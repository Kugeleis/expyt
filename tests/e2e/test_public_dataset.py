
# import e2e_server fixture from test_api_e2e


def test_e2e_public_energy_dataset(e2e_server: str) -> None:
    import httpx

    with httpx.Client(base_url=e2e_server) as client:
        # Use a dummy energy dataset imitating a public one
        csv_content = (
            b"country,energy_type,consumption,production,year\n"
            b"USA,Solar,100.5,102.0,2020\n"
            b"USA,Solar,105.0,108.5,2021\n"
            b"USA,Wind,50.0,52.0,2020\n"
            b"USA,Wind,55.5,58.0,2021\n"
            b"Germany,Solar,80.0,85.0,2020\n"
            b"Germany,Solar,85.0,90.5,2021\n"
            b"Germany,Wind,120.0,125.0,2020\n"
            b"Germany,Wind,130.5,135.0,2021\n"
        )

        files = {"file": ("energy_data.csv", csv_content, "text/csv")}
        resp = client.post("/wizard/upload", files=files)
        assert resp.status_code == 200
        assert resp.json()["id"] == "energy_data"

        # Create Session
        resp = client.post("/wizard/sessions")
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        # Step 1: Select dataset
        resp = client.post(
            f"/wizard/sessions/{session_id}/dataset",
            json={
                "dataset_id": "energy_data",
                "group_column": "country",
                "selected_value_columns": ["consumption", "production"],
            },
        )
        assert resp.status_code == 200

        # Step 2: Configure filters
        resp = client.post(
            f"/wizard/sessions/{session_id}/filters",
            json={"filters_config": []},
        )
        assert resp.status_code == 200

        # Step 3: Choose method
        resp = client.post(
            f"/wizard/sessions/{session_id}/method",
            json={"selected_method": "ttest_ind"},
        )
        assert resp.status_code == 200

        # Step 4: Run statistical evaluation
        resp = client.get(f"/wizard/sessions/{session_id}/results")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2  # two columns tested
        assert results[0]["method_name"] == "ttest_ind"

        # Step 5: Generate plots
        resp = client.post(
            f"/wizard/sessions/{session_id}/plots",
            json={"selected_plots": ["boxplot"], "top_n_columns": 2},
        )
        assert resp.status_code == 200

        # Check that 2 plots were generated (1 per column)
        assert len(resp.json()["plot_results"]) == 2

        # Step 6: Export results as JSON
        resp = client.post(
            f"/wizard/sessions/{session_id}/export",
            json={"export_format": "json"},
        )
        assert resp.status_code == 200
