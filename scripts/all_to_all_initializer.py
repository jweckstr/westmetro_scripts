import sys
from research.westmetro_paper.scripts.all_to_all import AllToAllRoutingPipeline
from research.westmetro_paper.scripts.util import split_into_equal_length_parts

if __name__ == "__main__":
    _, slurm_array_i, slurm_array_length = sys.argv
    slurm_array_i = int(slurm_array_i)
    slurm_array_length = int(slurm_array_length)

    assert(slurm_array_i < slurm_array_length)
    nodes = pandas.read_csv(HELSINKI_NODES_FNAME)['stop_I'].values
    parts = split_into_equal_length_parts(nodes, slurm_array_length)
    targets = parts[slurm_array_i]

    a2a_pipeline = AllToAllRoutingPipeline
    a2a_pipeline.loop_trough_targets_and_run_routing(targets)