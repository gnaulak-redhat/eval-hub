Feature: Status Endpoint
  As a service consumer
  I want to get service status information
  So that I can verify service details

  Scenario: Get service status
    Given the service is running
    When I send a GET request to "/api/v1/status"
    Then the response code should be 200
    And the response should be JSON
    And the response should contain "service" with value "eval-hub"
    And the response should contain "version" with value "1.0.0"
    And the response should contain "status" with value "running"
    And the response should contain "timestamp"

  Scenario: Status endpoint rejects non-GET methods
    Given the service is running
    When I send a POST request to "/api/v1/status"
    Then the response code should be 405
