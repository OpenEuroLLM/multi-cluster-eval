import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest
import yaml

from oellm.interactive_csv_builder import build_csv_interactive


class TestInteractiveCSVBuilder:
    """Test suite for the interactive CSV builder."""

    @pytest.fixture
    def temp_output_path(self):
        """Create a temporary output path for testing."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=True) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_basic_csv_creation(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test basic CSV creation with one model and one task."""
        # Mock user interactions
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",  # Choose to add a model
            "✅ Continue to tasks",  # Continue to tasks
            "➕ Add a single task",  # Add a task
            "0 (zero-shot)",  # Choose n_shot value
            "✅ Continue to preview",  # Continue to preview
        ]

        mock_text.return_value.ask.side_effect = [
            "test-model",  # Enter model name
            "test-task",  # Enter task name
        ]

        mock_confirm.return_value.ask.return_value = True  # Confirm save

        # Run the builder
        build_csv_interactive(temp_output_path)

        # Verify CSV was created
        assert Path(temp_output_path).exists()

        # Load and verify content
        df = pd.read_csv(temp_output_path)
        assert len(df) == 1
        assert df.iloc[0]["model_path"] == "test-model"
        assert df.iloc[0]["task_path"] == "test-task"
        assert df.iloc[0]["n_shot"] == 0

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_multiple_models_and_tasks(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test CSV creation with multiple models and tasks."""
        # Mock user interactions
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "➕ Add a model",
            "✅ Continue to tasks",
            "➕ Add a single task",
            "0,5 (both)",  # Multiple n_shot values
            "➕ Add a single task",
            "5 (few-shot)",
            "✅ Continue to preview",
        ]

        mock_text.return_value.ask.side_effect = [
            "model1",
            "meta-llama/Llama-2-7b-hf",
            "task1",
            "task2",
        ]

        mock_confirm.return_value.ask.return_value = True

        # Run the builder
        build_csv_interactive(temp_output_path)

        # Load and verify content
        df = pd.read_csv(temp_output_path)
        assert len(df) == 6  # 2 models × (2 n_shots for task1 + 1 n_shot for task2)

        # Check all combinations exist
        assert set(df["model_path"].unique()) == {"model1", "meta-llama/Llama-2-7b-hf"}
        assert set(df["task_path"].unique()) == {"task1", "task2"}

        # Check n_shot values for task1
        task1_df = df[df["task_path"] == "task1"]
        assert set(task1_df["n_shot"].unique()) == {0, 5}

        # Check n_shot values for task2
        task2_df = df[df["task_path"] == "task2"]
        assert set(task2_df["n_shot"].unique()) == {5}

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_custom_n_shot_values(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test custom n_shot value input."""
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "➕ Add a single task",
            "📝 Custom values",  # Choose custom n_shot
            "✅ Continue to preview",
        ]

        mock_text.return_value.ask.side_effect = [
            "test-model",
            "test-task",
            "0,3,7,15",  # Custom n_shot values
        ]

        mock_confirm.return_value.ask.return_value = True

        build_csv_interactive(temp_output_path)

        df = pd.read_csv(temp_output_path)
        assert len(df) == 4
        assert set(df["n_shot"].unique()) == {0, 3, 7, 15}

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_local_path_model(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test adding a model via local path."""
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "➕ Add a single task",
            "0 (zero-shot)",
            "✅ Continue to preview",
        ]

        mock_text.return_value.ask.side_effect = [
            "/path/to/local/model",  # Enter local path as model
            "test-task",  # Enter task name
        ]
        mock_confirm.return_value.ask.return_value = True

        build_csv_interactive(temp_output_path)

        df = pd.read_csv(temp_output_path)
        assert df.iloc[0]["model_path"] == "/path/to/local/model"

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_user_cancellation(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test user cancellation at various points."""
        # Test cancellation during model input
        mock_select.return_value.ask.return_value = None  # Simulate Ctrl+C

        build_csv_interactive(temp_output_path)

        # CSV should not be created
        assert not Path(temp_output_path).exists()

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_no_save_confirmation(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test when user chooses not to save."""
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "➕ Add a single task",
            "0 (zero-shot)",
            "✅ Continue to preview",
        ]

        mock_text.return_value.ask.side_effect = [
            "test-model",
            "test-task",
        ]

        mock_confirm.return_value.ask.return_value = False  # Don't save

        build_csv_interactive(temp_output_path)

        # CSV should not be created
        assert not Path(temp_output_path).exists()

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_invalid_n_shot_values(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test handling of invalid n_shot values."""
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "➕ Add a single task",
            "📝 Custom values",
            "➕ Add a single task",  # Add another task after invalid input
            "0 (zero-shot)",
            "✅ Continue to preview",
        ]

        mock_text.return_value.ask.side_effect = [
            "test-model",
            "test-task1",
            "invalid,values",  # Invalid n_shot values
            "test-task2",
        ]

        mock_confirm.return_value.ask.return_value = True

        build_csv_interactive(temp_output_path)

        df = pd.read_csv(temp_output_path)
        # Only the second task should be in the CSV
        assert len(df) == 1
        assert df.iloc[0]["task_path"] == "test-task2"

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_view_current_models_and_tasks(
        self, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test viewing current models and tasks functionality."""
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "📋 View current models",  # View models
            "✅ Continue to tasks",
            "➕ Add a single task",
            "0 (zero-shot)",
            "📋 View current tasks",  # View tasks
            "✅ Continue to preview",
        ]

        mock_text.return_value.ask.side_effect = [
            "test-model",
            "test-task",
        ]

        mock_confirm.return_value.ask.return_value = True

        # This should run without errors
        build_csv_interactive(temp_output_path)

        df = pd.read_csv(temp_output_path)
        assert len(df) == 1

    def test_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "output.csv"

            with patch(
                "oellm.interactive_csv_builder.questionary.select"
            ) as mock_select, patch(
                "oellm.interactive_csv_builder.questionary.text"
            ) as mock_text, patch(
                "oellm.interactive_csv_builder.questionary.confirm"
            ) as mock_confirm:
                mock_select.return_value.ask.side_effect = [
                    "➕ Add a model",
                    "✅ Continue to tasks",
                    "➕ Add a single task",
                    "0 (zero-shot)",
                    "✅ Continue to preview",
                ]

                mock_text.return_value.ask.side_effect = [
                    "test-model",
                    "test-task",
                ]

                mock_confirm.return_value.ask.return_value = True

                build_csv_interactive(str(nested_path))

                # Check that directory was created
                assert nested_path.parent.exists()
                assert nested_path.exists()

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.checkbox")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_single_task_group_selection(
        self,
        mock_file,
        mock_exists,
        mock_confirm,
        mock_checkbox,
        mock_select,
        temp_output_path,
    ):
        """Test selecting a single task group."""
        # Mock YAML content
        yaml_content = {
            "task_groups": {
                "open-sci-default": {
                    "description": "Default OpenEuroLLM scientific tasks",
                    "tasks": [
                        {"task": "copa", "n_shots": [0]},
                        {"task": "openbookqa", "n_shots": [0]},
                        {"task": "mmlu", "n_shots": [5]},
                    ],
                }
            }
        }
        mock_file.return_value.read.return_value = yaml.dump(yaml_content)
        mock_exists.return_value = True

        # Mock user interactions
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "📦 Use a default task group",
            "✅ Continue to preview",  # After adding task groups (line 201-208)
        ]

        mock_checkbox.return_value.ask.return_value = [
            "open-sci-default - Default OpenEuroLLM scientific tasks"
        ]

        # Mock text input for model
        with patch("oellm.interactive_csv_builder.questionary.text") as mock_text:
            mock_text.return_value.ask.return_value = "test-model"
            mock_confirm.return_value.ask.return_value = True

            build_csv_interactive(temp_output_path)

        # Verify CSV was created with correct content
        df = pd.read_csv(temp_output_path)
        assert len(df) == 3  # 3 tasks from the group
        assert set(df["task_path"]) == {"copa", "openbookqa", "mmlu"}
        assert df[df["task_path"] == "copa"]["n_shot"].values[0] == 0
        assert df[df["task_path"] == "mmlu"]["n_shot"].values[0] == 5

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.checkbox")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_multiple_task_groups_selection(
        self,
        mock_file,
        mock_exists,
        mock_confirm,
        mock_checkbox,
        mock_select,
        temp_output_path,
    ):
        """Test selecting multiple task groups."""
        # Mock YAML content with multiple groups
        yaml_content = {
            "task_groups": {
                "group1": {
                    "description": "First group",
                    "tasks": [
                        {"task": "task1", "n_shots": [0]},
                        {"task": "task2", "n_shots": [5]},
                    ],
                },
                "group2": {
                    "description": "Second group",
                    "tasks": [
                        {"task": "task3", "n_shots": [0, 5]},
                        {"task": "task4", "n_shots": [10]},
                    ],
                },
            }
        }
        mock_file.return_value.read.return_value = yaml.dump(yaml_content)
        mock_exists.return_value = True

        # Mock user interactions
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "📦 Use a default task group",
            "✅ Continue to preview",  # After adding task groups (line 201-208)
        ]

        mock_checkbox.return_value.ask.return_value = [
            "group1 - First group",
            "group2 - Second group",
        ]

        # Mock text input for model
        with patch("oellm.interactive_csv_builder.questionary.text") as mock_text:
            mock_text.return_value.ask.return_value = "test-model"
            mock_confirm.return_value.ask.return_value = True

            build_csv_interactive(temp_output_path)

        # Verify CSV was created with correct content
        df = pd.read_csv(temp_output_path)
        assert len(df) == 5  # 2 + 3 (task3 has 2 n_shots)
        assert set(df["task_path"]) == {"task1", "task2", "task3", "task4"}

        # Check n_shot values
        assert df[df["task_path"] == "task1"]["n_shot"].values[0] == 0
        assert df[df["task_path"] == "task2"]["n_shot"].values[0] == 5
        assert set(df[df["task_path"] == "task3"]["n_shot"].values) == {0, 5}
        assert df[df["task_path"] == "task4"]["n_shot"].values[0] == 10

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.checkbox")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_task_group_no_selection(
        self,
        mock_file,
        mock_exists,
        mock_confirm,
        mock_text,
        mock_checkbox,
        mock_select,
        temp_output_path,
    ):
        """Test when user opens task group menu but doesn't select any."""
        # Mock YAML content
        yaml_content = {
            "task_groups": {
                "group1": {
                    "description": "Test group",
                    "tasks": [{"task": "task1", "n_shots": [0]}],
                }
            }
        }
        mock_file.return_value.read.return_value = yaml.dump(yaml_content)
        mock_exists.return_value = True

        # Mock user interactions
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "📦 Use a default task group",
            "➕ Add a single task",  # Continue to add single task after no selection
            "0 (zero-shot)",
            "✅ Continue to preview",
        ]

        mock_checkbox.return_value.ask.return_value = []  # No groups selected

        mock_text.return_value.ask.side_effect = [
            "test-model",
            "manual-task",
        ]

        mock_confirm.return_value.ask.return_value = True

        build_csv_interactive(temp_output_path)

        # Verify CSV only contains the manually added task
        df = pd.read_csv(temp_output_path)
        assert len(df) == 1
        assert df["task_path"].values[0] == "manual-task"
        assert df["n_shot"].values[0] == 0

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    @patch("pathlib.Path.exists")
    def test_task_group_yaml_not_found(
        self, mock_exists, mock_confirm, mock_text, mock_select, temp_output_path
    ):
        """Test behavior when task-groups.yaml file doesn't exist."""
        # Mock that the YAML file doesn't exist
        mock_exists.return_value = False

        # Mock user interactions - no task group option should appear
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "➕ Add a single task",  # No task group option available
            "0 (zero-shot)",
            "✅ Continue to preview",
        ]

        mock_text.return_value.ask.side_effect = [
            "test-model",
            "test-task",
        ]

        mock_confirm.return_value.ask.return_value = True

        build_csv_interactive(temp_output_path)

        # Verify CSV was created with manually added task
        df = pd.read_csv(temp_output_path)
        assert len(df) == 1
        assert df["model_path"].values[0] == "test-model"
        assert df["task_path"].values[0] == "test-task"
        assert df["n_shot"].values[0] == 0

    @patch("oellm.interactive_csv_builder.questionary.select")
    @patch("oellm.interactive_csv_builder.questionary.checkbox")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    @patch("pathlib.Path.exists")
    @patch("builtins.open", new_callable=mock_open)
    def test_task_group_combined_with_individual_tasks(
        self,
        mock_file,
        mock_exists,
        mock_confirm,
        mock_text,
        mock_checkbox,
        mock_select,
        temp_output_path,
    ):
        """Test combining task groups with individually added tasks."""
        # Mock YAML content
        yaml_content = {
            "task_groups": {
                "small-group": {
                    "description": "Small test group",
                    "tasks": [
                        {"task": "group-task1", "n_shots": [0]},
                        {"task": "group-task2", "n_shots": [5]},
                    ],
                }
            }
        }
        mock_file.return_value.read.return_value = yaml.dump(yaml_content)
        mock_exists.return_value = True

        # Mock user interactions
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "✅ Continue to tasks",
            "📦 Use a default task group",
            "➕ Add more tasks",  # Choose to add more after task group
            "➕ Add a single task",
            "0,5 (both)",
            "➕ Add a single task",
            "📝 Custom values",
            "✅ Continue to preview",
        ]

        mock_checkbox.return_value.ask.return_value = ["small-group - Small test group"]

        mock_text.return_value.ask.side_effect = [
            "test-model",
            "individual-task1",
            "individual-task2",
            "0,10,25",  # Custom n_shot values
        ]

        mock_confirm.return_value.ask.return_value = True

        build_csv_interactive(temp_output_path)

        # Verify CSV contains both group tasks and individual tasks
        df = pd.read_csv(temp_output_path)

        # Should have: 2 group tasks + 2 individual-task1 n_shots + 3 individual-task2 n_shots = 7
        assert len(df) == 7

        # Check all tasks are present
        assert set(df["task_path"]) == {
            "group-task1",
            "group-task2",
            "individual-task1",
            "individual-task2",
        }

        # Verify n_shot values for each task
        assert df[df["task_path"] == "group-task1"]["n_shot"].values[0] == 0
        assert df[df["task_path"] == "group-task2"]["n_shot"].values[0] == 5
        assert set(df[df["task_path"] == "individual-task1"]["n_shot"].values) == {0, 5}
        assert set(df[df["task_path"] == "individual-task2"]["n_shot"].values) == {
            0,
            10,
            25,
        }
