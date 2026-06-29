from __future__ import annotations

import os
import re
import socket
import threading
import time
from collections.abc import Generator

import pandas as pd
import playwright.sync_api
import pytest
import uvicorn
from playwright.sync_api import Dialog, Page, expect

from app.main import app


def _check_playwright_launch() -> bool:
    try:
        with playwright.sync_api.sync_playwright() as p:
            browser = p.chromium.launch()
            browser.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _check_playwright_launch(),
    reason="Playwright chromium browser cannot be launched (missing system dependencies)",
)


def get_free_port() -> int:
    """Get a free port on localhost."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


@pytest.fixture(scope="module")
def playwright_server(tmp_path_factory: pytest.TempPathFactory) -> Generator[str, None, None]:
    """Start uvicorn server in a background thread for playwright tests."""
    tmp_dir = tmp_path_factory.mktemp("playwright_data")

    # Save a test dataset in it
    df = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "value": [10.0, 10.5, 11.0, 10.2, 9.8, 12.0, 12.5, 13.0, 12.2, 11.8],
            "value2": [5.0, 5.5, 6.0, 5.2, 4.8, 7.0, 7.5, 8.0, 7.2, 6.8],
        }
    )
    df.to_csv(tmp_dir / "playwright_normal_data.csv", index=False)

    df3 = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
            "value": [10.0, 10.5, 11.0, 10.2, 9.8, 12.0, 12.5, 13.0, 12.2, 11.8, 14.0, 14.5, 15.0, 14.2, 13.8],
        }
    )
    df3.to_csv(tmp_dir / "playwright_three_groups.csv", index=False)

    df_mixed = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "value": [10.0, 10.5, 11.0, 10.2, 9.8, 12.0, 12.5, 13.0, 12.2, 11.8],
            "category": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "Y"],
        }
    )
    df_mixed.to_csv(tmp_dir / "playwright_mixed_data.csv", index=False)

    original_env = os.environ.get("EXPYRI_DATA_DIR")
    os.environ["EXPYRI_DATA_DIR"] = str(tmp_dir)

    port = get_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Playwright test server failed to start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)

    if original_env is not None:
        os.environ["EXPYRI_DATA_DIR"] = original_env
    else:
        os.environ.pop("EXPYRI_DATA_DIR", None)


def test_playwright_restart_session(playwright_server: str, page: Page) -> None:
    """E2E Playwright test verifying that restarting session via UI dialog resets the state."""
    # 1. Open the wizard
    page.goto(f"{playwright_server}/")

    # Assert title is present
    expect(page.locator("h2")).to_have_text("Experiment Evaluation Wizard")

    # Get the initial restart button hx-post value to extract session ID
    restart_btn = page.locator(".btn-restart")
    expect(restart_btn).to_be_visible()

    hx_post = restart_btn.get_attribute("hx-post")
    assert hx_post is not None
    match = re.search(r"/sessions/([a-f0-9]+)/restart", hx_post)
    assert match is not None
    session_id_1 = match.group(1)

    # 2. Select the dataset to put the session in a dirty state
    select_locator = page.locator("select[name='dataset_id']")
    select_locator.select_option("playwright_normal_data")

    # Wait for the UI to update and show column selection
    page.wait_for_selector("select[name='group_column']")

    # 3. Setup dialog handler to automatically accept the confirmation restart dialog
    dialog_handled = False

    def handle_dialog(dialog: Dialog) -> None:
        nonlocal dialog_handled
        assert "restart" in dialog.message.lower()
        dialog_handled = True
        dialog.accept()

    page.on("dialog", handle_dialog)

    # 4. Click the Restart Session button
    restart_btn.click()

    # 5. Wait for redirection back to the root step (Dataset Selection)
    page.wait_for_url(f"{playwright_server}/")

    # Wait for the restart button to be visible again on the new page
    page.wait_for_selector(".btn-restart")

    # 6. Verify that a new session ID was generated
    new_hx_post = page.locator(".btn-restart").get_attribute("hx-post")
    assert new_hx_post is not None
    new_match = re.search(r"/sessions/([a-f0-9]+)/restart", new_hx_post)
    assert new_match is not None
    session_id_2 = new_match.group(1)

    assert session_id_1 != session_id_2
    assert dialog_handled, "Confirmation dialog was not intercepted"

    # 7. Verify that dataset select has returned to the default state (no dataset selected)
    expect(page.locator("select[name='dataset_id']")).to_have_value("")


def test_playwright_full_wizard_flow(playwright_server: str, page: Page) -> None:
    """E2E Playwright test executing a complete UI wizard flow from Step 1 to Step 6."""
    # 1. Open the wizard homepage
    page.goto(f"{playwright_server}/")
    expect(page.locator("h2")).to_have_text("Experiment Evaluation Wizard")

    # Step 1: Select dataset
    page.locator("select[name='dataset_id']").select_option("playwright_normal_data")
    page.wait_for_selector("select[name='group_column']")

    # Configure group column
    page.locator("select[name='group_column']").select_option("group")
    page.wait_for_selector("input[name='selected_groups']")

    # Step 2: Skip/Continue past optional Filters step via sidebar navigation
    page.click("text=Choose Method")
    page.wait_for_selector("input[name='selected_method']")

    # Step 3: Choose method
    page.locator("input[name='selected_method'][value='ttest_ind']").check()

    # Step 4: Go to Results
    page.click("text=Results")
    page.wait_for_selector("input[id='plots-sig-filter']")

    # Assert results are displayed
    expect(page.locator("table")).to_contain_text("ttest_ind")
    expect(page.locator("body")).to_contain_text("value")

    # Step 5: Go to Visualizations
    page.click("text=Select Plots")
    page.wait_for_selector("input[name='selected_plots']")

    # Check the boxplot checkbox
    page.locator("input[name='selected_plots'][value='boxplot']").check()

    # Step 6: Go to Export
    page.click("text=Export")
    page.wait_for_selector("text=PDF Report")

    # Assert export choices are visible
    expect(page.locator("text=PDF Report")).to_be_visible()
    expect(page.locator("text=JSON Schema")).to_be_visible()
    expect(page.locator("text=CSV Dataset")).to_be_visible()


def test_playwright_filters_next_button_enabled(playwright_server: str, page: Page) -> None:
    """E2E Playwright test verifying that the Continue button is enabled on Step 2.

    Checks that it's enabled even if no filters are configured.
    """
    page.goto(f"{playwright_server}/")

    # Select dataset and group column
    page.locator("select[name='dataset_id']").select_option("playwright_normal_data")
    page.wait_for_selector("select[name='group_column']")
    page.locator("select[name='group_column']").select_option("group")
    page.wait_for_selector("input[name='selected_groups']")

    # Navigate to Step 2 (Configure Filters)
    page.click("text=Configure Filters")
    page.wait_for_selector("text=Step 2: Preprocessing Filters")

    # Verify the Continue button is enabled
    next_btn = page.locator("#btn-sidebar-next")
    expect(next_btn).to_be_enabled()


def test_playwright_method_applicability(playwright_server: str, page: Page) -> None:
    """E2E Playwright test verifying that statistical methods are enabled/disabled correctly on Step 3.

    Checks that ANOVA is enabled and t-test is disabled for a 3-group dataset.
    """
    page.goto(f"{playwright_server}/")

    # Select dataset with 3 groups
    page.locator("select[name='dataset_id']").select_option("playwright_three_groups")
    page.wait_for_selector("select[name='group_column']")
    page.locator("select[name='group_column']").select_option("group")
    page.wait_for_selector("input[name='selected_groups']")

    # Navigate to Step 3 (Choose Method)
    page.click("text=Choose Method")
    page.wait_for_selector("input[name='selected_method']")

    # Assert that anova is enabled (since groups = 3)
    anova_radio = page.locator("input[name='selected_method'][value='anova']")
    expect(anova_radio).to_be_enabled()

    # Assert that ttest_ind is disabled (since groups = 3, and t-test requires exactly 2 groups)
    ttest_radio = page.locator("input[name='selected_method'][value='ttest_ind']")
    expect(ttest_radio).to_be_disabled()


def test_playwright_method_enables_next_button(playwright_server: str, page: Page) -> None:
    """E2E Playwright test verifying that selecting a statistical method enables the sidebar next button.

    Checks that it's disabled initially and enabled after selecting a method.
    """
    page.goto(f"{playwright_server}/")

    # Select dataset and group column
    page.locator("select[name='dataset_id']").select_option("playwright_mixed_data")
    page.wait_for_selector("select[name='group_column']")
    page.locator("select[name='group_column']").select_option("group")
    page.wait_for_selector("input[name='selected_groups']")

    # Navigate to Step 3 (Choose Method)
    page.click("text=Choose Method")
    page.wait_for_selector("input[name='selected_method']")

    # Verify the Continue button is disabled initially
    next_btn = page.locator("#btn-sidebar-next")
    expect(next_btn).to_be_disabled()

    # Check a method (e.g. ttest_ind)
    page.locator("input[name='selected_method'][value='ttest_ind']").check()

    # Verify the Continue button becomes enabled
    expect(next_btn).to_be_enabled()

    # Click the next button
    next_btn.click()

    # Verify we transition to Step 4: Results
    page.wait_for_selector("text=Step 4: Statistical Results")


def test_playwright_dataset_selection_requires_group_column(playwright_server: str, page: Page) -> None:
    """E2E Playwright test verifying that dataset selection alone keeps the next button disabled.

    Checks that selecting a group column enables it.
    """
    page.goto(f"{playwright_server}/")

    # Select dataset
    page.locator("select[name='dataset_id']").select_option("playwright_normal_data")
    page.wait_for_selector("select[name='group_column']")

    # Verify next button is disabled
    next_btn = page.locator("#btn-sidebar-next")
    expect(next_btn).to_be_disabled()

    # Select group column
    page.locator("select[name='group_column']").select_option("group")
    page.wait_for_selector("input[name='selected_groups']")

    # Verify next button becomes enabled
    expect(next_btn).to_be_enabled()


def test_playwright_visualization_cards(playwright_server: str, page: Page) -> None:
    """E2E Playwright test verifying that generating plots keeps us on Step 5 and groups plots by dependent column."""
    page.goto(f"{playwright_server}/")

    # Step 1: Select dataset and group column
    page.locator("select[name='dataset_id']").select_option("playwright_normal_data")
    page.wait_for_selector("select[name='group_column']")
    page.locator("select[name='group_column']").select_option("group")
    page.wait_for_selector("input[name='selected_groups']")

    # Ensure both value and value2 continuous variables are selected
    page.locator("input[name='selected_value_columns'][value='value']").check()
    page.locator("input[name='selected_value_columns'][value='value2']").check()

    # Step 2: Skip filters
    page.click("text=Configure Filters")
    page.wait_for_selector("text=Step 2: Preprocessing Filters")

    # Step 3: Select method
    page.click("text=Choose Method")
    page.wait_for_selector("input[name='selected_method']")
    page.locator("input[name='selected_method'][value='ttest_ind']").check()

    # Step 4: Run results
    page.click("#btn-sidebar-next")
    page.wait_for_selector("text=Step 4: Statistical Results")

    # Step 5: Go to Visualizations
    page.click("#btn-sidebar-next")
    page.wait_for_selector("text=Step 5: Visualizations")

    # Check both boxplot and violin checkboxes
    page.locator("input[name='selected_plots'][value='boxplot']").check()
    page.locator("input[name='selected_plots'][value='violinplot']").check()

    # Click Generate Plots
    page.click("#btn-generate-plots")

    # Verify we STAY on Step 5: Visualizations
    page.wait_for_selector("text=Step 5: Visualizations")

    # Verify that we have exactly 2 cards (article elements) inside #plots-display
    # because we visualized 2 columns: value and value2
    cards = page.locator("#plots-display article")
    expect(cards).to_have_count(2)

    # Verify each card contains statistical results in header
    expect(cards.nth(0).locator("header")).to_contain_text("p:")
    expect(cards.nth(1).locator("header")).to_contain_text("p:")

    # Verify each card contains exactly 2 images (one boxplot, one violin)
    first_card_images = cards.nth(0).locator("img")
    expect(first_card_images).to_have_count(2)

    second_card_images = cards.nth(1).locator("img")
    expect(second_card_images).to_have_count(2)
