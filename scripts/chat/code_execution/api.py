import os
import time
import json
import signal
import logging
import argparse
import asyncio
import sys
import tornado.ioloop
import tornado.web
import tornado.httpserver
from collections import namedtuple
from jupyter import JupyterKernel, JupyterGatewayDocker, JupyterGatewayKubernetes

logging.basicConfig(level=logging.INFO)

if os.environ.get("USE_KUBERNETES", "0").lower() == "1":
    JupyterKernelWrapper = JupyterGatewayKubernetes
    logging.info("Using Kubernetes as the backend for JupyterGateway")
else:
    JupyterKernelWrapper = JupyterGatewayDocker
    logging.info("Using Docker as the backend for JupyterGateway")

# Global data structure to map convid to (JupyterKernelWrapper, JupyterKernel)
JupyterKernelType = namedtuple("JupyterKernelType", [
    "kernel_wrapper",
    "kernel",
    "last_access_time"
])


async def initialize_conversation_kernel(app, convid):
    """Create a single kernel, cleaning up a gateway if initialization fails."""
    kernel_wrapper = JupyterKernelWrapper(name=f"conv-{convid}")
    entered = False
    try:
        url_suffix = kernel_wrapper.__enter__()
        entered = True
        if os.environ.get("DEBUG", False):
            logging.info(f"Kernel URL: {url_suffix}")
        kernel = JupyterKernel(url_suffix, convid)
        await kernel.initialize()
        app.conv_id_to_kernel[convid] = JupyterKernelType(
            kernel_wrapper, kernel, time.time()
        )
        logging.info(f"Kernel created for conversation {convid}")
    except BaseException:
        if entered:
            try:
                kernel_wrapper.__exit__(*sys.exc_info())
            except Exception:
                logging.exception("Failed to close kernel gateway after initialization error")
        raise


async def ensure_conversation_kernel(app, convid):
    """Coalesce concurrent first requests for one conversation, not all sessions."""
    if convid in app.conv_id_to_kernel:
        return False

    pending = app.kernel_initializations
    task = pending.get(convid)
    created = task is None
    if task is None:
        task = asyncio.create_task(initialize_conversation_kernel(app, convid))
        pending[convid] = task

        def forget_finished(done):
            if pending.get(convid) is done:
                pending.pop(convid, None)

        task.add_done_callback(forget_finished)

    # Canceling one waiting HTTP request must not abort its peer's setup.
    await asyncio.shield(task)
    return created

def cleanup_kernels(app, force=False):
    """Cleanup kernels and gateway dockers that have timed out."""
    KERNEL_TIMEOUT = 10 * 60  # 10 minutes
    current_time = time.time()
    to_delete = []
    conv_id_to_kernel = app.conv_id_to_kernel
    # Find all kernels that have timed out
    for convid in conv_id_to_kernel.keys():
        last_access = conv_id_to_kernel[convid].last_access_time
        if current_time - last_access > KERNEL_TIMEOUT:
            to_delete.append(convid)

    if force:
        to_delete = list(conv_id_to_kernel.keys())
        logging.info(f"Force cleanup all {len(to_delete)} kernels")

    for convid in to_delete:
        # Close the kernel
        # kernel: JupyterKernel = conv_id_to_kernel[convid].kernel
        # kernel.shutdown()  # Close the JupyterKernel
        # Close the JupyterKernelWrapper by close its context manager
        kernel_wrapper = conv_id_to_kernel[convid].kernel_wrapper
        kernel_wrapper.__exit__(None, None, None)  # Close the JupyterKernelWrapper
        # Delete the entry from the global data structure
        del conv_id_to_kernel[convid]
        logging.info(f"Kernel closed for conversation {convid}")

class ExecuteHandler(tornado.web.RequestHandler):
    async def post(self):
        data = json.loads(self.request.body)
        convid = data.get("convid")
        code = data.get("code")

        conv_id_to_kernel = self.application.conv_id_to_kernel
        new_kernel = await ensure_conversation_kernel(self.application, convid)

        # Update last access time
        kernel_access_time = time.time()
        conv_id_to_kernel[convid] = conv_id_to_kernel[convid]._replace(
            last_access_time=kernel_access_time
        )

        # Execute the code
        kernel: JupyterKernel = conv_id_to_kernel[convid].kernel
        result = await kernel.execute(code)

        self.write(json.dumps({
            "result": result,
            "new_kernel_created": new_kernel
        }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = tornado.web.Application([
        (r"/execute", ExecuteHandler),
        # Add other routes here
    ])
    app.conv_id_to_kernel = {}
    app.kernel_initializations = {}
    
    # Wrap cleanup_kernels to pass the app object
    periodic_cleanup = tornado.ioloop.PeriodicCallback(
        lambda: cleanup_kernels(app),
        int(os.environ.get("CLEANUP_TIMEOUT_MS", 60000))
    )
    periodic_cleanup.start()

    # Setup signal handler
    def signal_handler(signum, frame, app):
        logging.info("Received SIGINT, cleaning up...")
        cleanup_kernels(app, force=True)
        tornado.ioloop.IOLoop.current().stop()
        logging.info("Cleanup complete, shutting down.")

    signal.signal(
        signal.SIGINT,
        lambda signum, frame: signal_handler(signum, frame, app)
    )
    server = tornado.httpserver.HTTPServer(app)
    server.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
