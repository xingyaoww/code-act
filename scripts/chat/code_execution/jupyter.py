import os
import time
import docker
import asyncio
import tornado
from tornado.escape import json_encode, json_decode, url_escape
from tornado.websocket import websocket_connect, WebSocketHandler
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from uuid import uuid4

# from predefine_tools import (
#     GOOGLE_SEARCH,
#     GET_URL_CONTENT
# )

class JupyterKernel:
    def __init__(
        self,
        url_suffix,
        convid,
        lang="python"
    ):
        self.base_url = f"http://{url_suffix}"
        self.base_ws_url = f"ws://{url_suffix}"
        self.lang = lang
        self.kernel_id = None
        self.ws = None
        self.convid = convid
        print(f"Jupyter kernel created for conversation {convid}")

        self.heartbeat_interval = 10000  # 10 seconds
        self.heartbeat_callback = None

    async def initialize(self):
        await self.execute(r"%colors nocolor")
        # pre-defined tools
        self.tools_to_run = [
            # TODO: You can add code for your pre-defined tools here (need to improve this)
        ]
        for tool in self.tools_to_run:
            # print(f"Tool initialized:\n{tool}")
            await self.execute(tool)
    
    async def _send_heartbeat(self):
        if not self.ws:
            return
        try:
            self.ws.ping()
            # print("Heartbeat sent...")
        except tornado.iostream.StreamClosedError:
            # print("Heartbeat failed, reconnecting...")
            try:
                await self._connect()
            except ConnectionRefusedError:
                print("ConnectionRefusedError: Failed to reconnect to kernel websocket - Is the kernel still running?")

    async def _connect(self):
        if self.ws:
            self.ws.close()
            self.ws = None

        client = AsyncHTTPClient()
        if not self.kernel_id:
            response = await client.fetch(
                "{}/api/kernels".format(self.base_url),
                method="POST",
                body=json_encode({"name": self.lang}),
            )
            kernel = json_decode(response.body)
            self.kernel_id = kernel["id"]

        ws_req = HTTPRequest(
            url="{}/api/kernels/{}/channels".format(
                self.base_ws_url, url_escape(self.kernel_id)
            )
        )
        self.ws = await websocket_connect(ws_req)
        print("Connected to kernel websocket")

        # Setup heartbeat
        if self.heartbeat_callback:
            self.heartbeat_callback.stop()
        self.heartbeat_callback = PeriodicCallback(self._send_heartbeat, self.heartbeat_interval)
        self.heartbeat_callback.start()

    async def execute(self, code, timeout=60):
        if not self.ws:
            await self._connect()

        msg_id = uuid4().hex
        self.ws.write_message(
            json_encode(
                {
                    "header": {
                        "username": "",
                        "version": "5.0",
                        "session": "",
                        "msg_id": msg_id,
                        "msg_type": "execute_request",
                    },
                    "parent_header": {},
                    "channel": "shell",
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                    },
                    "metadata": {},
                    "buffers": {},
                }
            )
        )

        outputs = []


        async def wait_for_messages():
            execution_done = False
            while not execution_done:
                msg = await self.ws.read_message()
                msg = json_decode(msg)
                msg_type = msg['msg_type']
                parent_msg_id = msg['parent_header'].get('msg_id', None)

                if parent_msg_id != msg_id:
                    continue

                if os.environ.get("DEBUG", False):
                    print(f"MSG TYPE: {msg_type.upper()} DONE:{execution_done}\nCONTENT: {msg['content']}")

                if msg_type == 'error':
                    traceback = "\n".join(msg["content"]["traceback"])
                    outputs.append(traceback)
                    execution_done = True
                elif msg_type == 'stream':
                    outputs.append(msg['content']['text'])
                elif msg_type in ['execute_result', 'display_data']:
                    outputs.append(msg['content']['data']['text/plain'])
                    if 'image/png' in msg['content']['data']:
                        # use markdone to display image (in case of large image)
                        # outputs.append(f"\n<img src=\"data:image/png;base64,{msg['content']['data']['image/png']}\"/>\n")
                        outputs.append(f"![image](data:image/png;base64,{msg['content']['data']['image/png']})")

                elif msg_type == 'execute_reply':
                    execution_done = True
            return execution_done

        async def interrupt_kernel():
            client = AsyncHTTPClient()
            interrupt_response = await client.fetch(
                f"{self.base_url}/api/kernels/{self.kernel_id}/interrupt",
                method="POST",
                body=json_encode({"kernel_id": self.kernel_id}),
            )
            print(f"Kernel interrupted: {interrupt_response}")

        try:
            execution_done = await asyncio.wait_for(wait_for_messages(), timeout)
        except asyncio.TimeoutError:
            await interrupt_kernel()
            return f"[Execution timed out ({timeout} seconds).]"

        if not outputs and execution_done:
            ret = "[Code executed successfully with no output]"
        else:
            ret = ''.join(outputs)

        if os.environ.get("DEBUG", False):
            print(f"OUTPUT:\n{ret}")
        return ret

    async def shutdown_async(self):
        if self.kernel_id:
            client = AsyncHTTPClient()
            await client.fetch(
                "{}/api/kernels/{}".format(self.base_url, self.kernel_id),
                method="DELETE",
            )
            self.kernel_id = None
            if self.ws:
                self.ws.close()
                self.ws = None

class JupyterGatewayDocker:
    DOCKER_IMAGE = "docker.io/xingyaoww/agent-jupyter-kernel-gateway"
    RESOURCE_CONSTRAINTS = {
        'mem_limit': '8g', # Limit to 8 GB of memory
        'nano_cpus': 2 * 10 ** 9  # Limit to 2 CPU cores
        # Add other constraints as needed
    }

    def __init__(self):
        self.client = docker.from_env()
        self.container = None
        self.url = None

    def _get_free_port(self):
        """Get a free port from the OS."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    def _wait_for_container(self, container):
        # Wait for a specific log message or condition indicating the app inside is ready
        # Note: This is just an example, you should adjust the condition to match your use case
        timeout = 60  # seconds
        start_time = time.time()
        while True:
            # Get the latest logs
            logs = container.logs().decode('utf-8')
            print(logs)
            # Check for a specific message or condition in the logs
            if "Jupyter Kernel Gateway" in logs and "is available at" in logs:
                break
            # Exit if timeout reached
            if time.time() - start_time > timeout:
                print("Timeout reached while waiting for container to be ready.")
                break
            time.sleep(1)

    def __enter__(self):
        # Check if the image exists locally, if not, pull it
        try:
            self.client.images.get(self.DOCKER_IMAGE)
        except docker.errors.ImageNotFound:
            self.client.images.pull(self.DOCKER_IMAGE)

        port = self._get_free_port()
        # Run the container and expose the port
        self.container = self.client.containers.run(
            self.DOCKER_IMAGE,
            detach=True,
            ports={"8888/tcp": port},
            remove=True,  # Removes container when it's stopped
            **self.RESOURCE_CONSTRAINTS,
        )
        self._wait_for_container(self.container)

        url_suffix = f"localhost:{port}"
        return url_suffix

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.container:
            self.container.stop()


if __name__ == "__main__":

    TO_EXEC = """
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
results = search.results("python", num_results=1)

# Obtain the link from the first search result
search_link = results[0]['link']
print("Search Result Link:", search_link)
"""
    TO_EXEC_1 = """
from boilerpy3 import extractors
extractor = extractors.ArticleExtractor()

# From a URL
content = extractor.get_content_from_url(search_link)
# If you want HTML (can be messy)
# content = extractor.get_marked_html_from_url(search_link)

# Display the first line of the extracted content
first_line = content.splitlines()[0]
print("First Line from Search Result:", first_line)
"""

    TO_EXEC_2 = """
from langchain.utilities import GoogleSearchAPIWrapper

search_elon = GoogleSearchAPIWrapper()
results_elon = search_elon.results("Elon Musk birth year", num_results=1)
print(results_elon)

search_taylor = GoogleSearchAPIWrapper()
results_taylor = search_taylor.results("Taylor Swift birth year", num_results=1)
print(results_taylor)

search_messi = GoogleSearchAPIWrapper()
results_messi = search_messi.results("Lionel Messi birth year", num_results=1)
print(results_messi)

def calculate_age(birth_year, current_year=2023):
    return current_year - birth_year
ages = {
    "Elon Musk": calculate_age(1971),
    "Taylor Swift": calculate_age(1989),
    "Lionel Messi": calculate_age(1987)
}
ages
"""
    with JupyterGatewayDocker() as url_suffix:
        j = JupyterKernel(url_suffix)
        print("EXECUTE:")
        print(TO_EXEC)
        print("OUTPUT:")
        print(j.execute(TO_EXEC))

        print("EXECUTE:")
        print(TO_EXEC_1)
        print("OUTPUT:")
        print(j.execute(TO_EXEC_1))

        print("EXECUTE:")
        print(TO_EXEC_2)
        print("OUTPUT:")
        print(j.execute(TO_EXEC_2))

        # print(j.execute((
        #     "prime_numbers = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]\n"
        #     "mean_prime = sum(prime_numbers) / len(prime_numbers)\n"
        #     "mean_prime\n"
        # )))
        # print(j.execute("print('Hello from Jupyter!')"))
        # print(j.execute("print(a)"))
        # print(j.execute("a = 1 * 9"))
        # print(j.execute("print(a)"))
        # print(j.execute("import transformers"))
        # print(j.execute("!pwd"))
        # print(j.execute("!pip3 install transformers"))
        del j
