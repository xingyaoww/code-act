import time
import json
import signal
import argparse
import tornado.ioloop
import tornado.web
import tornado.httpserver
from collections import namedtuple
from jupyter import JupyterKernel, JupyterGatewayDocker

# Global data structure to map convid to (JupyterGatewayDocker, JupyterKernel)
JupyterKernelType = namedtuple("JupyterKernelType", [
    "gateway_docker",
    "kernel",
    "last_access_time"
])

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
        print(f"Force cleanup all {len(to_delete)} kernels")

    for convid in to_delete:
        # Close the kernel
        # kernel: JupyterKernel = conv_id_to_kernel[convid].kernel
        # kernel.shutdown()  # Close the JupyterKernel
        # Close the JupyterGatewayDocker by close its context manager
        gateway_docker = conv_id_to_kernel[convid].gateway_docker
        gateway_docker.__exit__(None, None, None)  # Close the JupyterGatewayDocker
        # Delete the entry from the global data structure
        del conv_id_to_kernel[convid]
        print(f"Kernel closed for conversation {convid}")

class ExecuteHandler(tornado.web.RequestHandler):
    async def post(self):
        data = json.loads(self.request.body)
        convid = data.get("convid")
        code = data.get("code")

        # Create a new kernel if not exist
        new_kernel = False

        conv_id_to_kernel = self.application.conv_id_to_kernel
        if convid not in conv_id_to_kernel:
            gateway_docker = JupyterGatewayDocker()
            url_suffix = gateway_docker.__enter__()
            kernel = JupyterKernel(url_suffix, convid)
            await kernel.initialize()
            conv_id_to_kernel[convid] = JupyterKernelType(
                gateway_docker,
                kernel,
                None
            )
            new_kernel = True
            print(f"Kernel created for conversation {convid}")

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
    
    # Wrap cleanup_kernels to pass the app object
    periodic_cleanup = tornado.ioloop.PeriodicCallback(
        lambda: cleanup_kernels(app),
        60000
    )
    periodic_cleanup.start()

    # Setup signal handler
    def signal_handler(signum, frame, app):
        print("Received SIGINT, cleaning up...")
        cleanup_kernels(app, force=True)
        tornado.ioloop.IOLoop.current().stop()
        print("Cleanup complete, shutting down.")

    signal.signal(
        signal.SIGINT,
        lambda signum, frame: signal_handler(signum, frame, app)
    )
    server = tornado.httpserver.HTTPServer(app)
    server.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
