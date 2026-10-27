name: Pull Request
about: Submit a pull request
title: '[PR] '
labels: ''
assignees: ''

body:
  - type: markdown
    attributes:
      value: |
        Thanks for submitting a pull request! Please fill out the checklist below.

  - type: checkboxes
    id: checklist
    attributes:
      label: Checklist
      options:
        - label: I have read the [Contributing Guidelines](CONTRIBUTING.md)
          required: true
        - label: I have run `make test` and all tests pass
          required: true
        - label: I have run `make lint` and there are no linting errors
          required: true
        - label: I have run `make typecheck` and there are no type errors
          required: false
        - label: I have updated the documentation if needed
          required: true
        - label: I have added tests for any new functionality
          required: true
        - label: I have updated the CHANGELOG.md
          required: false

  - type: textarea
    id: description
    attributes:
      label: Description
      description: A clear and concise description of what this PR does.
    validations:
      required: true

  - type: textarea
    id: related-issues
    attributes:
      label: Related Issues
      description: List any related issues (e.g., 'Closes #123', 'Fixes #456').

  - type: textarea
    id: breaking-changes
    attributes:
      label: Breaking Changes
      description: Does this PR introduce any breaking changes? If yes, please describe them and the migration path.

  - type: textarea
    id: additional-notes
    attributes:
      label: Additional Notes
      description: Add any additional notes, screenshots, or context for the reviewers.
