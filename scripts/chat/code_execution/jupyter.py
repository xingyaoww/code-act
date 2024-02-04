import os
import time
import docker
import asyncio
import tornado
import logging

from kubernetes import client
from kubernetes import config
from tornado.escape import json_encode, json_decode, url_escape
from tornado.websocket import websocket_connect, WebSocketHandler
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from uuid import uuid4

# from predefine_tools import (
#     GOOGLE_SEARCH,
#     GET_URL_CONTENT
# )

logging.basicConfig(level=logging.INFO)
config.load_incluster_config()


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
        logging.info(f"Jupyter kernel created for conversation {convid} at {url_suffix}")

        self.heartbeat_interval = 10000  # 10 seconds
        self.heartbeat_callback = None

    async def initialize(self):
        await self.execute(r"%colors nocolor")
        # pre-defined tools
        self.tools_to_run = [
            # TODO: You can add code for your pre-defined tools here (need to improve this)
        ]
        for tool in self.tools_to_run:
            # logging.info(f"Tool initialized:\n{tool}")
            await self.execute(tool)
    
    async def _send_heartbeat(self):
        if not self.ws:
            return
        try:
            self.ws.ping()
            # logging.info("Heartbeat sent...")
        except tornado.iostream.StreamClosedError:
            # logging.info("Heartbeat failed, reconnecting...")
            try:
                await self._connect()
            except ConnectionRefusedError:
                logging.info("ConnectionRefusedError: Failed to reconnect to kernel websocket - Is the kernel still running?")

    async def _connect(self):
        if self.ws:
            self.ws.close()
            self.ws = None

        client = AsyncHTTPClient()
        if not self.kernel_id:
            n_tries = 5
            while n_tries > 0:
                try:
                    response = await client.fetch(
                        "{}/api/kernels".format(self.base_url),
                        method="POST",
                        body=json_encode({"name": self.lang}),
                    )
                    kernel = json_decode(response.body)
                    self.kernel_id = kernel["id"]
                    break
                except Exception as e:
                    # kernels are not ready yet
                    n_tries -= 1
                    await asyncio.sleep(1)
            
            if n_tries == 0:
                raise ConnectionRefusedError("Failed to connect to kernel")

        ws_req = HTTPRequest(
            url="{}/api/kernels/{}/channels".format(
                self.base_ws_url, url_escape(self.kernel_id)
            )
        )
        self.ws = await websocket_connect(ws_req)
        logging.info("Connected to kernel websocket")

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
                    logging.info(f"MSG TYPE: {msg_type.upper()} DONE:{execution_done}\nCONTENT: {msg['content']}")

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
            logging.info(f"Kernel interrupted: {interrupt_response}")

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
            logging.info(f"OUTPUT:\n{ret}")
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
    DOCKER_IMAGE = "docker.io/xingyaoww/codeact-executor"
    RESOURCE_CONSTRAINTS = {
        'mem_limit': '8g', # Limit to 8 GB of memory
        'nano_cpus': 2 * 10 ** 9  # Limit to 2 CPU cores
        # Add other constraints as needed
    }

    def __init__(self, name: str):
        self.name = name
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
            logging.info(logs)
            # Check for a specific message or condition in the logs
            if "Jupyter Kernel Gateway" in logs and "is available at" in logs:
                break
            # Exit if timeout reached
            if time.time() - start_time > timeout:
                logging.info("Timeout reached while waiting for container to be ready.")
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
            name=self.name,
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


class JupyterGatewayKubernetes:
    IMAGE = "docker.io/xingyaoww/codeact-executor"
    NAMESPACE = os.environ.get("KUBERNETES_NAMESPACE", "codeact-chat-ui")
    RESOURCE_CONSTRAINTS = {
        'limits': {'memory': "512Mi", 'cpu': "200m"},
        'requests': {'memory': "256Mi", 'cpu': "100m"}
    }

    def __init__(self, name):
        self.pod_name = f"executor-{name}-{uuid4().hex[:6]}"  # Generate a unique pod name
        self.api_instance = client.CoreV1Api()
        self.port = 8888  # Default port for Jupyter Kernel Gateway

    def _create_pod(self):
        # Define container
        container = client.V1Container(
            name=self.pod_name,
            image=self.IMAGE,
            ports=[client.V1ContainerPort(container_port=self.port)],
            resources=client.V1ResourceRequirements(
                **self.RESOURCE_CONSTRAINTS
            ),
        )

        # Define pod spec
        metadata = client.V1ObjectMeta(
            name=self.pod_name,
            labels={"app": self.pod_name}  # Add this line to include labels
        )
        spec = client.V1PodSpec(
            containers=[container],
            restart_policy="Never"
        )
        pod = client.V1Pod(
            api_version="v1",
            kind="Pod",
            metadata=metadata,
            spec=spec,
        )

        # Create pod
        self.api_instance.create_namespaced_pod(
            namespace=self.NAMESPACE,
            body=pod
        )
        logging.info(f"Pod {self.pod_name} is being created...")

    def _create_service(self):
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(
                name=self.pod_name
            ),
            spec=client.V1ServiceSpec(
                selector={"app": self.pod_name},
                ports=[client.V1ServicePort(port=self.port, target_port=self.port)],
                type="ClusterIP"  # Default and suitable for internal communication
            )
        )
        self.api_instance.create_namespaced_service(namespace=self.NAMESPACE, body=service)
        logging.info(f"Service {self.pod_name} created.")


    def _wait_for_pod_to_be_ready(self):
        ready = False
        while not ready:
            pod_status = self.api_instance.read_namespaced_pod_status(
                self.pod_name, self.NAMESPACE
            )
            if pod_status.status.phase == "Running":
                ready = True
                # self.pod_ip = pod_status.status.pod_ip
                logging.info(f"Pod Ready. IP: {pod_status.status.pod_ip}")
            else:
                time.sleep(1)  # Wait and check again

    def __enter__(self):
        config.load_incluster_config()  # Make sure your kubeconfig is correctly set up
        self._create_pod()
        self._wait_for_pod_to_be_ready()
        self._create_service()  # Ensure the service is created after the pod is ready
        service = self.api_instance.read_namespaced_service(
            name=self.pod_name, namespace=self.NAMESPACE
        )
        service_ip = service.spec.cluster_ip
        return f"{service_ip}:{self.port}"

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.api_instance.delete_namespaced_service(name=self.pod_name, namespace=self.NAMESPACE)
        logging.info(f"Service {self.pod_name} deleted.")
        self.api_instance.delete_namespaced_pod(self.pod_name, self.NAMESPACE)
        logging.info(f"Pod {self.pod_name} has been deleted.")


if __name__ == "__main__":

    TO_EXEC = """
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()
results = search.results("python", num_results=1)

# Obtain the link from the first search result
search_link = results[0]['link']
logging.info("Search Result Link:", search_link)
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
logging.info("First Line from Search Result:", first_line)
"""

    TO_EXEC_2 = """
from langchain.utilities import GoogleSearchAPIWrapper

search_elon = GoogleSearchAPIWrapper()
results_elon = search_elon.results("Elon Musk birth year", num_results=1)
logging.info(results_elon)

search_taylor = GoogleSearchAPIWrapper()
results_taylor = search_taylor.results("Taylor Swift birth year", num_results=1)
logging.info(results_taylor)

search_messi = GoogleSearchAPIWrapper()
results_messi = search_messi.results("Lionel Messi birth year", num_results=1)
logging.info(results_messi)

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
        logging.info("EXECUTE:")
        logging.info(TO_EXEC)
        logging.info("OUTPUT:")
        logging.info(j.execute(TO_EXEC))

        logging.info("EXECUTE:")
        logging.info(TO_EXEC_1)
        logging.info("OUTPUT:")
        logging.info(j.execute(TO_EXEC_1))

        logging.info("EXECUTE:")
        logging.info(TO_EXEC_2)
        logging.info("OUTPUT:")
        logging.info(j.execute(TO_EXEC_2))

        # logging.info(j.execute((
        #     "prime_numbers = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]\n"
        #     "mean_prime = sum(prime_numbers) / len(prime_numbers)\n"
        #     "mean_prime\n"
        # )))
        # logging.info(j.execute("logging.info('Hello from Jupyter!')"))
        # logging.info(j.execute("logging.info(a)"))
        # logging.info(j.execute("a = 1 * 9"))
        # logging.info(j.execute("logging.info(a)"))
        # logging.info(j.execute("import transformers"))
        # logging.info(j.execute("!pwd"))
        # logging.info(j.execute("!pip3 install transformers"))
        del j
