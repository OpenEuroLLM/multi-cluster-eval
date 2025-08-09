import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

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
            "📝 Custom input",  # Choose custom input for model
            "✅ Continue to tasks",  # Continue to tasks
            "➕ Add a task",  # Add a task
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
            "📝 Custom input",
            "➕ Add a model",
            "🤗 HuggingFace model (e.g., meta-llama/Llama-2-7b-hf)",
            "✅ Continue to tasks",
            "➕ Add a task",
            "0,5 (both)",  # Multiple n_shot values
            "➕ Add a task",
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
            "📝 Custom input",
            "✅ Continue to tasks",
            "➕ Add a task",
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
    @patch("oellm.interactive_csv_builder.questionary.path")
    @patch("oellm.interactive_csv_builder.questionary.text")
    @patch("oellm.interactive_csv_builder.questionary.confirm")
    def test_local_path_model(
        self, mock_confirm, mock_text, mock_path, mock_select, temp_output_path
    ):
        """Test adding a model via local path."""
        mock_select.return_value.ask.side_effect = [
            "➕ Add a model",
            "📁 Local path",  # Choose local path
            "✅ Continue to tasks",
            "➕ Add a task",
            "0 (zero-shot)",
            "✅ Continue to preview",
        ]

        mock_path.return_value.ask.return_value = "/path/to/local/model"
        mock_text.return_value.ask.return_value = "test-task"
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
            "📝 Custom input",
            "✅ Continue to tasks",
            "➕ Add a task",
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
            "📝 Custom input",
            "✅ Continue to tasks",
            "➕ Add a task",
            "📝 Custom values",
            "➕ Add a task",  # Add another task after invalid input
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
            "📝 Custom input",
            "📋 View current models",  # View models
            "✅ Continue to tasks",
            "➕ Add a task",
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
                    "📝 Custom input",
                    "✅ Continue to tasks",
                    "➕ Add a task",
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
