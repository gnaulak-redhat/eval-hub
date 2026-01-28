Feature: Evaluations Endpoint
  As a data scientist
  I want to create evaluation jobs
  So that I evaluate models

  Scenario: Create an evaluation job
    Given the service is running
    When I send a POST request to "/api/v1/evaluations/jobs" with body "file:/evaluation_job.json"
    Then the response code should be 202
