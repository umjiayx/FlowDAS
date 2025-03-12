#!/bin/bash

cd ../

# Clean up after running all experiments
echo "Cleaning up directories..."
rm -rf data*
rm -rf runs_*
rm -rf __pycache__
echo "Cleanup complete!"