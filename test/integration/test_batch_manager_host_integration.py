import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from inference.manager import BatchManager, wrapper
from config.prompts import SystemRole


@pytest.fixture
async def mock_host_and_manager():
    with patch("inference.host.Host", autospec=True) as MockHost:
        mock_host_instance = MockHost.return_value
        mock_host_instance.__aenter__.return_value = mock_host_instance

        # Mock the process_request method of Host
        mock_host_instance.process_request.return_value = AsyncMock(
            side_effect=[
                {"choices": [{"message": {"content": "Processed content 1."}}]},
                {"choices": [{"message": {"content": "Processed content 2."}}]},
            ]
        )

        manager = BatchManager(mock_host_instance, batch_size=2, max_wait_time=1.0)
        runner_task = asyncio.create_task(manager.run())
        yield mock_host_instance, manager

        # Teardown: ensure manager stops and runner task is cancelled
        manager.running = False
        if not runner_task.done():
            runner_task.cancel()
            try:
                await runner_task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_batch_manager_host_integration(mock_host_and_manager):
    mock_host_instance, manager = mock_host_and_manager

    url1 = "http://example.com/item1"
    text1 = "Text for item 1"
    role1 = SystemRole.summary

    url2 = "http://example.com/item2"
    text2 = "Text for item 2"
    role2 = SystemRole.summary

    task1 = asyncio.create_task(wrapper(url1, role1, text1, manager))
    task2 = asyncio.create_task(wrapper(url2, role2, text2, manager))

    results = await asyncio.gather(task1, task2)

    # Check results
    assert len(results) == 2
    assert results[0] == (
        url1,
        role1,
        {"choices": [{"message": {"content": "Processed content 1."}}]},
    )
    assert results[1] == (
        url2,
        role2,
        {"choices": [{"message": {"content": "Processed content 2."}}]},
    )

    # Verify that Host.process_request was called correctly
    mock_host_instance.process_request.assert_has_calls(
        [
            AsyncMock(url=url1, role=role1, input_text=text1),
            AsyncMock(url=url2, role=role2, input_text=text2),
        ],
        any_order=True,
    )
    assert mock_host_instance.process_request.call_count == 2

    # Verify that manager's run task completed or was cancelled cleanly
    # (This is handled by the fixture's teardown, so no explicit assert here)
