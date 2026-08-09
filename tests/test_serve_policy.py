from __future__ import annotations

import logging
import threading
import unittest

import numpy as np
import zmq

from serve_policy import (
    ACTION_STEPS,
    ContractError,
    BackendOutputError,
    FakePolicyBackend,
    NumpyMsgpackCodec,
    PolicyApplication,
    STATE_KEY,
    TASK_KEY,
    THIRD_VIEW_KEY,
    WRENCH_KEY,
    WRIST_KEY,
    ZmqPolicyServer,
    validate_observation,
)


def valid_observation():
    return {
        STATE_KEY: np.zeros((8,), dtype=np.float32),
        WRENCH_KEY: np.zeros((6,), dtype=np.float32),
        WRIST_KEY: np.zeros((480, 640, 3), dtype=np.uint8),
        THIRD_VIEW_KEY: np.zeros((480, 640, 3), dtype=np.uint8),
        TASK_KEY: "insert the plug",
    }


def action_request(observation=None):
    return {
        "endpoint": "get_action",
        "data": {
            "observation": observation or valid_observation(),
            "options": None,
        },
    }


class CodecTests(unittest.TestCase):
    def test_numpy_round_trip(self):
        source = {
            "float": np.arange(8, dtype=np.float32),
            "image": np.arange(24, dtype=np.uint8).reshape(2, 4, 3),
            "scalar": np.float32(1.25),
        }
        restored = NumpyMsgpackCodec.loads(NumpyMsgpackCodec.dumps(source))
        np.testing.assert_array_equal(restored["float"], source["float"])
        np.testing.assert_array_equal(restored["image"], source["image"])
        self.assertEqual(restored["scalar"], 1.25)

    def test_object_arrays_are_forbidden(self):
        with self.assertRaisesRegex(TypeError, "object-dtype"):
            NumpyMsgpackCodec.dumps(np.asarray([object()], dtype=object))

    def test_pickle_npy_is_rejected(self):
        import io

        stream = io.BytesIO()
        np.save(stream, np.asarray([{"unsafe": True}], dtype=object), allow_pickle=True)
        envelope = {
            "__ndarray_class__": True,
            "as_npy": stream.getvalue(),
        }
        payload = __import__("msgpack").packb(envelope, use_bin_type=True)
        with self.assertRaises(ValueError):
            NumpyMsgpackCodec.loads(payload)


class ObservationContractTests(unittest.TestCase):
    def test_valid_observation(self):
        restored = validate_observation(valid_observation())
        self.assertEqual(tuple(restored), (
            STATE_KEY,
            WRENCH_KEY,
            WRIST_KEY,
            THIRD_VIEW_KEY,
            TASK_KEY,
        ))

    def test_missing_field_is_rejected(self):
        observation = valid_observation()
        del observation[WRENCH_KEY]
        with self.assertRaisesRegex(ContractError, "missing observation fields"):
            validate_observation(observation)

    def test_wrong_dtype_and_shape_are_rejected(self):
        observation = valid_observation()
        observation[STATE_KEY] = np.zeros((8,), dtype=np.float64)
        with self.assertRaisesRegex(ContractError, "dtype float32"):
            validate_observation(observation)

        observation = valid_observation()
        observation[WRIST_KEY] = np.zeros((640, 480, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ContractError, "shape"):
            validate_observation(observation)

    def test_empty_task_and_unknown_field_are_rejected(self):
        observation = valid_observation()
        observation[TASK_KEY] = "  "
        with self.assertRaisesRegex(ContractError, "non-empty"):
            validate_observation(observation)

        observation = valid_observation()
        observation["silently_ignored"] = np.zeros(1)
        with self.assertRaisesRegex(ContractError, "unknown observation fields"):
            validate_observation(observation)


class ApplicationTests(unittest.TestCase):
    def test_fake_backend_returns_16_by_d(self):
        backend = FakePolicyBackend(action_dim=8, action_steps=16)
        application = PolicyApplication(backend)
        response = application.handle(action_request())
        actions, info = response
        self.assertEqual(actions.shape, (16, 8))
        self.assertEqual(actions.dtype, np.float32)
        self.assertEqual(info["model_name"], "FACTE")
        self.assertEqual(info["action_shape"], [16, 8])
        self.assertIn("unused", info["observation_field_usage"][STATE_KEY])
        self.assertIn("no language encoder", info["observation_field_usage"][TASK_KEY])

    def test_long_chunk_is_truncated_from_the_end(self):
        backend = FakePolicyBackend(action_dim=8, action_steps=20)
        application = PolicyApplication(backend)
        actions, _ = application.handle(action_request())
        expected = backend.predict(valid_observation())[:ACTION_STEPS]
        np.testing.assert_array_equal(actions, expected)

    def test_short_chunk_is_an_error(self):
        application = PolicyApplication(FakePolicyBackend(action_dim=8, action_steps=15))
        with self.assertRaisesRegex(BackendOutputError, "at least 16"):
            application.handle(action_request())

    def test_reset_clears_all_episode_state(self):
        backend = FakePolicyBackend()
        application = PolicyApplication(backend)
        application.handle(action_request())
        self.assertEqual(len(backend.episode_state), 1)
        self.assertEqual(application.episode_request_count, 1)

        response = application.handle(
            {"endpoint": "reset", "data": {"options": None}}
        )
        self.assertEqual(response, {"status": "ok", "reset": True})
        self.assertEqual(backend.episode_state, [])
        self.assertEqual(application.episode_request_count, 0)


class ZmqRecoveryTests(unittest.TestCase):
    def setUp(self):
        logger = logging.getLogger("factr_policy_server_test")
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        self.backend = FakePolicyBackend()
        self.server = ZmqPolicyServer(
            PolicyApplication(self.backend),
            host="127.0.0.1",
            port=0,
            logger=logger,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.assertTrue(self.server.ready_event.wait(timeout=3.0))
        self.assertIsNotNone(self.server.bound_endpoint)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, 3000)
        self.socket.setsockopt(zmq.SNDTIMEO, 3000)
        self.socket.connect(self.server.bound_endpoint)

    def tearDown(self):
        self.socket.close(linger=0)
        self.context.term()
        self.server.stop()
        self.thread.join(timeout=3.0)
        self.assertFalse(self.thread.is_alive())

    def call(self, request):
        self.socket.send(NumpyMsgpackCodec.dumps(request))
        return NumpyMsgpackCodec.loads(self.socket.recv())

    def test_error_response_preserves_rep_state(self):
        invalid = action_request()
        invalid["data"]["observation"][STATE_KEY] = np.zeros((7,), dtype=np.float32)
        response = self.call(invalid)
        self.assertIn("error", response)

        ping = self.call({"endpoint": "ping"})
        self.assertEqual(ping["status"], "ok")

        actions, info = self.call(action_request())
        self.assertEqual(actions.shape, (16, 8))
        self.assertEqual(info["model_name"], "FACTE")

        reset = self.call({"endpoint": "reset", "data": {"options": None}})
        self.assertTrue(reset["reset"])
        self.assertEqual(self.backend.episode_state, [])

    def test_malformed_messagepack_preserves_rep_state(self):
        self.socket.send(b"\xc1")
        response = NumpyMsgpackCodec.loads(self.socket.recv())
        self.assertIn("error", response)
        ping = self.call({"endpoint": "ping"})
        self.assertEqual(ping["status"], "ok")


if __name__ == "__main__":
    unittest.main()
