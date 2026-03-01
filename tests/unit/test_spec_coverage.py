import requests
from scripts.test_scripts.test_spec_coverage import SpecTester


def test_substitute_path_params():
    """Test replacing path parameters with test_ prefix."""
    tester = SpecTester(spec_path="dummy.yaml")
    result = tester.substitute_path_params(
        "/repos/{owner}/{repo}/issues/{issue_number}"
    )
    assert result == "/repos/test_owner/test_repo/issues/test_issue_number"


def test_find_crud_pairs():
    """Test discovering paths that have both GET and POST operations."""
    tester = SpecTester(spec_path="dummy.yaml")
    tester.spec = {
        "paths": {
            "/api/items/{item_id}": {
                "get": {"operationId": "get_item"},
                "post": {"operationId": "create_item"},
            },
            "/api/info": {"get": {"operationId": "get_info"}},
        }
    }

    pairs = tester.find_crud_pairs()

    assert len(pairs) == 1
    assert pairs[0]["original_path"] == "/api/items/{item_id}"
    assert pairs[0]["endpoint"] == "/api/items/test_item_id"
    assert pairs[0]["post_op"] == {"operationId": "create_item"}
    assert pairs[0]["get_op"] == {"operationId": "get_item"}


def test_generate_payload():
    """Test payload generation with fallback strategies and $ref resolution."""
    tester = SpecTester(spec_path="dummy.yaml")
    tester.spec = {
        "components": {
            "schemas": {
                "NestedObject": {"type": "string", "example": "resolved_nested_value"}
            }
        }
    }

    schema = {
        "properties": {
            "prop_example": {"example": "hardcoded_example"},
            "prop_string": {"type": "string"},
            "prop_int": {"type": "integer"},
            "prop_ref": {"$ref": "#/components/schemas/NestedObject"},
        }
    }

    payload = tester.generate_payload(schema)

    assert payload["prop_example"] == "hardcoded_example"
    assert payload["prop_string"] == "test_prop_string_8832"
    assert payload["prop_int"] == 1
    assert payload["prop_ref"] == "resolved_nested_value"


def test_run_state_test_success(mocker):
    """Test state test where identifier is found in subsequent GET request."""
    tester = SpecTester(spec_path="dummy.yaml")

    mock_post_response = mocker.Mock()
    mock_post_response.status_code = 201

    mock_get_response = mocker.Mock()
    mock_get_response.status_code = 200
    mock_get_response.text = '{"name": "test_marker_8832"}'

    mocker.patch("requests.post", return_value=mock_post_response)
    mocker.patch("requests.get", return_value=mock_get_response)

    result = tester.run_state_test("/api/test", {"name": "test_marker_8832"})

    assert result is True
    requests.post.assert_called_once()
    requests.get.assert_called_once()


def test_run_state_test_failure(mocker):
    """Test state test where identifier is NOT found in subsequent GET request."""
    tester = SpecTester(spec_path="dummy.yaml")

    mock_post_response = mocker.Mock()
    mock_post_response.status_code = 201

    mock_get_response = mocker.Mock()
    mock_get_response.status_code = 200
    mock_get_response.text = '{"name": "other_value"}'

    mocker.patch("requests.post", return_value=mock_post_response)
    mocker.patch("requests.get", return_value=mock_get_response)

    result = tester.run_state_test("/api/test", {"name": "test_marker_8832"})

    assert result is False


def test_run_state_test_network_error(mocker):
    """Test state test where a connection error occurs."""
    tester = SpecTester(spec_path="dummy.yaml")

    mocker.patch(
        "requests.post",
        side_effect=requests.exceptions.ConnectionError("Connection Failed"),
    )

    result = tester.run_state_test("/api/test", {"name": "test_marker_8832"})

    assert result is False
