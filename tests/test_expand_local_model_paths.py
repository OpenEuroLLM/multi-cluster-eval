import tempfile
from pathlib import Path

import pytest

from oellm.main import _expand_local_model_paths


class TestExpandLocalModelPaths:
    """Test suite for the _expand_local_model_paths function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def create_safetensors_file(self, path: Path, name: str = "model.safetensors"):
        """Helper to create a dummy safetensors file."""
        path.mkdir(parents=True, exist_ok=True)
        (path / name).touch()

    def test_single_model_directory(self, temp_dir):
        """Test a directory containing safetensors files directly."""
        model_dir = temp_dir / "model"
        self.create_safetensors_file(model_dir)

        result = _expand_local_model_paths(str(model_dir))

        assert len(result) == 1
        assert result[0] == model_dir

    def test_model_with_hf_checkpoints(self, temp_dir):
        """Test a model with hf/iter_* checkpoint structure."""
        model_dir = temp_dir / "model"

        # Create checkpoint structure
        checkpoint1 = model_dir / "hf" / "iter_0001000"
        checkpoint2 = model_dir / "hf" / "iter_0002000"
        checkpoint3 = model_dir / "hf" / "iter_0003000"

        self.create_safetensors_file(checkpoint1)
        self.create_safetensors_file(checkpoint2)
        self.create_safetensors_file(checkpoint3)

        result = _expand_local_model_paths(str(model_dir))

        assert len(result) == 3
        assert checkpoint1 in result
        assert checkpoint2 in result
        assert checkpoint3 in result

    def test_directory_with_iteration_subdirs(self, temp_dir):
        """Test a directory directly containing iter_* subdirectories."""
        model_dir = temp_dir / "model_a"

        # Create iteration directories directly under model_a
        iter1 = model_dir / "iter_0001000"
        iter2 = model_dir / "iter_0002000"
        iter3 = model_dir / "iter_0003000"

        self.create_safetensors_file(iter1)
        self.create_safetensors_file(iter2)
        self.create_safetensors_file(iter3)

        result = _expand_local_model_paths(str(model_dir))

        assert len(result) == 3
        assert iter1 in result
        assert iter2 in result
        assert iter3 in result

    def test_directory_with_multiple_models(self, temp_dir):
        """Test a directory containing multiple model subdirectories."""
        parent_dir = temp_dir / "converted_checkpoints"

        # Create multiple models
        model1 = parent_dir / "open-sci-ref_model-0.13b_data-c4"
        model2 = parent_dir / "open-sci-ref_model-0.35b_data-c4"

        self.create_safetensors_file(model1)
        self.create_safetensors_file(model2)

        result = _expand_local_model_paths(str(parent_dir))

        assert len(result) == 2
        assert model1 in result
        assert model2 in result

    def test_multiple_models_with_checkpoints(self, temp_dir):
        """Test multiple models each with their own checkpoints."""
        parent_dir = temp_dir / "models"

        # Model 1 with checkpoints
        model1_checkpoint1 = parent_dir / "model1" / "hf" / "iter_1000"
        model1_checkpoint2 = parent_dir / "model1" / "hf" / "iter_2000"

        # Model 2 with checkpoints
        model2_checkpoint1 = parent_dir / "model2" / "hf" / "iter_1000"
        model2_checkpoint2 = parent_dir / "model2" / "hf" / "iter_2000"

        self.create_safetensors_file(model1_checkpoint1)
        self.create_safetensors_file(model1_checkpoint2)
        self.create_safetensors_file(model2_checkpoint1)
        self.create_safetensors_file(model2_checkpoint2)

        result = _expand_local_model_paths(str(parent_dir))

        assert len(result) == 4
        assert model1_checkpoint1 in result
        assert model1_checkpoint2 in result
        assert model2_checkpoint1 in result
        assert model2_checkpoint2 in result

    def test_empty_directory(self, temp_dir):
        """Test an empty directory returns no models."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()

        result = _expand_local_model_paths(str(empty_dir))

        assert len(result) == 0

    def test_non_existent_directory(self, temp_dir):
        """Test a non-existent directory returns no models."""
        non_existent = temp_dir / "does_not_exist"

        result = _expand_local_model_paths(str(non_existent))

        assert len(result) == 0

    def test_directory_with_non_model_files(self, temp_dir):
        """Test a directory with files but no safetensors."""
        dir_with_files = temp_dir / "not_a_model"
        dir_with_files.mkdir()
        (dir_with_files / "readme.txt").touch()
        (dir_with_files / "config.json").touch()

        result = _expand_local_model_paths(str(dir_with_files))

        assert len(result) == 0

    def test_mixed_structure(self, temp_dir):
        """Test a directory with mixed structure (some models, some checkpoints)."""
        parent_dir = temp_dir / "mixed"

        # Direct model
        direct_model = parent_dir / "direct_model"
        self.create_safetensors_file(direct_model)

        # Model with checkpoints
        checkpoint_model = parent_dir / "checkpoint_model" / "hf" / "iter_1000"
        self.create_safetensors_file(checkpoint_model)

        # Empty directory
        (parent_dir / "empty_dir").mkdir(parents=True)

        # Non-model files
        (parent_dir / "readme.txt").touch()

        result = _expand_local_model_paths(str(parent_dir))

        assert len(result) == 2
        assert direct_model in result
        assert checkpoint_model in result

    def test_file_instead_of_directory(self, temp_dir):
        """Test passing a file instead of a directory."""
        file_path = temp_dir / "file.txt"
        file_path.touch()

        result = _expand_local_model_paths(str(file_path))

        assert len(result) == 0

    def test_symlinked_directory(self, temp_dir: Path):
        """Test handling of symlinked directories."""
        # Create actual model directory
        actual_model = temp_dir / "actual_model"
        self.create_safetensors_file(actual_model)

        # Create symlink to model
        symlink = temp_dir / "symlinked_model"
        symlink.symlink_to(actual_model)

        result = _expand_local_model_paths(str(symlink))

        assert len(result) == 1
        assert result[0] == symlink  # Should return the symlink path, not the target
