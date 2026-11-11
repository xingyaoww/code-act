"""Deterministic concurrency regressions for the shipped Tornado code executor.

The Docker/Jupyter boundary is stubbed, while the actual ExecuteHandler and
per-conversation allocation path are exercised through concurrent HTTP calls.
No model, browser or external provider is invoked.
"""

import asyncio
import json
import sys
import types
from unittest.mock import patch

import tornado.testing
import tornado.web

# The production jupyter module loads Docker/Kubernetes integration at import.
# Keep the native Tornado handler while replacing only the external kernel.
stub_jupyter = types.ModuleType("jupyter")
stub_jupyter.JupyterKernel = object
stub_jupyter.JupyterGatewayDocker = object
stub_jupyter.JupyterGatewayKubernetes = object
with patch.dict(sys.modules, {"jupyter": stub_jupyter}):
    import api


class FakeGateway:
    created = []
    closed = []

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.created.append(self)
        return self.name

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed.append(self)


class FakeKernel:
    entered = None
    two_entered = None
    release = None
    initialize_count = 0

    def __init__(self, url_suffix, convid):
        self.convid = convid

    async def initialize(self):
        self.__class__.initialize_count += 1
        self.__class__.entered.set()
        if self.__class__.initialize_count >= 2:
            self.__class__.two_entered.set()
        await self.__class__.release.wait()

    async def execute(self, code):
        return {"convid": self.convid, "code": code}


class ObservedHandler(api.ExecuteHandler):
    async def post(self):
        self.application.requests_started += 1
        if self.application.requests_started >= 2:
            self.application.two_requests_started.set()
        await super().post()


class CodeExecutorConcurrencyTest(tornado.testing.AsyncHTTPTestCase):
    def get_app(self):
        app = tornado.web.Application([(r"/execute", ObservedHandler)])
        app.conv_id_to_kernel = {}
        app.kernel_initializations = {}
        app.requests_started = 0
        app.two_requests_started = asyncio.Event()
        return app

    def setUp(self):
        super().setUp()
        FakeGateway.created = []
        FakeGateway.closed = []
        FakeKernel.initialize_count = 0
        FakeKernel.entered = asyncio.Event()
        FakeKernel.two_entered = asyncio.Event()
        FakeKernel.release = asyncio.Event()
        gateway_patch = patch.object(api, "JupyterKernelWrapper", FakeGateway)
        kernel_patch = patch.object(api, "JupyterKernel", FakeKernel)
        gateway_patch.start()
        kernel_patch.start()
        self.addCleanup(gateway_patch.stop)
        self.addCleanup(kernel_patch.stop)

    def request(self, convid):
        return self.http_client.fetch(
            self.get_url("/execute"),
            method="POST",
            body=json.dumps({"convid": convid, "code": "1 + 1"}),
        )

    @tornado.testing.gen_test(timeout=10)
    async def test_same_conversation_first_requests_share_one_kernel(self):
        first = self.request("shared")
        await asyncio.wait_for(FakeKernel.entered.wait(), 3)
        second = self.request("shared")
        await asyncio.wait_for(self._app.two_requests_started.wait(), 3)
        await asyncio.sleep(0)
        FakeKernel.release.set()

        responses = await asyncio.gather(first, second)
        self.assertTrue(all(response.code == 200 for response in responses))
        self.assertEqual(len(FakeGateway.created), 1, "concurrent first requests must not create two gateways")
        self.assertEqual(len(self._app.conv_id_to_kernel), 1)
        flags = [json.loads(r.body)["new_kernel_created"] for r in responses]
        self.assertEqual(sorted(flags), [False, True])
        self.assertEqual(FakeGateway.closed, [])

    @tornado.testing.gen_test(timeout=10)
    async def test_distinct_conversations_initialize_concurrently(self):
        first = self.request("one")
        await asyncio.wait_for(FakeKernel.entered.wait(), 3)
        second = self.request("two")
        await asyncio.wait_for(FakeKernel.two_entered.wait(), 3)
        FakeKernel.release.set()
        responses = await asyncio.gather(first, second)
        self.assertTrue(all(r.code == 200 for r in responses))
        self.assertEqual(len(FakeGateway.created), 2)
        self.assertEqual(len(self._app.conv_id_to_kernel), 2)
