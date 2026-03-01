import yaml
from scripts.test_generator import TestGenerator as Generator


def test_endpoint_extraction_success(tmp_path):
    spec_data = {"paths": {"/repos/{owner}": {"get": {}, "post": {}}}}
    dummy_file = tmp_path / "dummy_spec.yaml"
    with open(dummy_file, "w") as f:
        yaml.dump(spec_data, f)

    generator = Generator(spec_path=str(dummy_file))

    expected_endpoints = ["GET /repos/{owner}", "POST /repos/{owner}"]
    assert sorted(generator.api_endpoints) == sorted(expected_endpoints)


def test_endpoint_extraction_fallback():
    generator = Generator(spec_path="non_existent_file.yaml")
    assert generator.api_endpoints == ["GET /", "POST /repos/{owner}/{repo}/issues"]


def test_test_case_generation_happy_path(mocker):
    mocker.patch(
        "scripts.test_generator.generate_response", return_value="dummy response"
    )

    mock_valid_cases = [
        {
            "method": "GET",
            "path": "/",
            "headers": {},
            "body": None,
            "description": "test",
        }
    ]
    mocker.patch(
        "scripts.test_generator.clean_llm_response", return_value=mock_valid_cases
    )

    generator = Generator(spec_path="non_existent_file.yaml")
    result = generator.generate_test_cases(n=1, provider="gemini", model="test")

    assert result == mock_valid_cases


def test_test_case_generation_error(mocker):
    mocker.patch(
        "scripts.test_generator.generate_response", return_value="dummy response"
    )
    mocker.patch(
        "scripts.test_generator.clean_llm_response", return_value={"error": "Failed"}
    )

    generator = Generator(spec_path="non_existent_file.yaml")
    result = generator.generate_test_cases(n=1, provider="gemini", model="test")

    assert result == []


def test_test_case_generation_malformed_keys(mocker):
    mocker.patch(
        "scripts.test_generator.generate_response", return_value="dummy response"
    )

    mock_invalid_cases = [
        {"method": "GET", "headers": {}, "body": None, "description": "test"}
    ]
    mocker.patch(
        "scripts.test_generator.clean_llm_response", return_value=mock_invalid_cases
    )

    generator = Generator(spec_path="non_existent_file.yaml")
    result = generator.generate_test_cases(n=1, provider="gemini", model="test")

    assert result == []
