"""Generate the Pseudo-GT blendshape coefficients
"""
import argparse
import numpy as np
import os
from tqdm import tqdm
from said.util.parser import parse_list
from said.optimize.blendshape_coeffs import OptimizationProblemFull
from dataset.dataset_voca import BlendVOCADataset, BlendVOCAPseudoGTOptDataset


def main():
    """Main function"""

    parser = argparse.ArgumentParser(
        description="Generate the Pseudo-GT blendshape coefficients by solving the optimization problem"
    )
    parser.add_argument(
        "--neutrals_dir",
        type=str,
        default="./BlendVOCA/templates_head",
        help="Directory of the neutral meshes",
    )
    parser.add_argument(
        "--blendshapes_dir",
        type=str,
        default="./BlendVOCA/blendshapes_head",
        help="Directory of the blendshape meshes",
    )
    parser.add_argument(
        "--mesh_seqs_dir",
        type=str,
        default="./BlendVOCA/unposedcleaneddata",
        help="Directory of the mesh sequences",
    )
    parser.add_argument(
        "--blendshape_list_path",
        type=str,
        default= "./data/ARKit_blendshapes.txt",
        help="List of the blendshapes",
    )
    parser.add_argument(
        "--head_idx_path",
        type=str,
        default= "./data/FLAME_head_idx.txt",
        help="List of the head indices. Empty string will disable this option.",
    )
    parser.add_argument(
        "--blendshapes_coeffs_out_dir",
        type=str,
        default="./BlendVOCA/bs_npy",
        help="Directory of the output coefficients",
    )
    args = parser.parse_args()

    neutrals_dir = args.neutrals_dir
    blendshapes_dir = args.blendshapes_dir
    mesh_seqs_dir = args.mesh_seqs_dir

    blendshape_list_path = args.blendshape_list_path
    head_idx_path = args.head_idx_path

    blendshapes_coeffs_out_dir = args.blendshapes_coeffs_out_dir

    def coeff_out_path(person_id: str, seq_id: int, exists_ok: bool = False) -> str:
        """Generate the output path of the coefficients.
        If you want to change the output file name, then change this function

        Parameters
        ----------
        person_id : str
            Person id
        seq_id : int
            Sequence id
        exists_ok : bool, optional
            If false, raise error when the file already exists, by default False

        Returns
        -------
        str
            Output path of the coefficients
        """
        out_dir = blendshapes_coeffs_out_dir
        os.makedirs(out_dir,exist_ok=True)

        out_path = os.path.join(out_dir, f"{person_id}_sentence{seq_id:02}.npy")

        return out_path

    # Parse blendshape name
    blendshape_name_list = parse_list(blendshape_list_path, str)

    # Parse head indices
    head_idx_list = None if head_idx_path == "" else parse_list(head_idx_path, int)

    dataset = BlendVOCAPseudoGTOptDataset(
        neutrals_dir, blendshapes_dir, mesh_seqs_dir, blendshape_name_list
    )

    person_id_list = (
        BlendVOCADataset.person_ids_train
        + BlendVOCADataset.person_ids_val
        + BlendVOCADataset.person_ids_test
    )
    seq_id_list = BlendVOCADataset.sentence_ids

    for person_id in tqdm(person_id_list):
        bl_out = dataset.get_blendshapes(person_id)

        neutral_mesh = bl_out.neutral
        blendshapes_meshes_dict = bl_out.blendshapes

        neutral_vertices = neutral_mesh.vertices
        blendshapes_vertices_list = [
            blendshapes_meshes_dict[name].vertices for name in blendshape_name_list
        ]

        neutral_vector = neutral_vertices.reshape((-1, 1))
        blendshape_vectors = []
        for v in blendshapes_vertices_list:
            blendshape_vectors.append(v.reshape((-1, 1)))

        blendshapes_matrix = np.concatenate(blendshape_vectors, axis=1)

        # Define the optimization problem
        opt_prob = OptimizationProblemFull(neutral_vector, blendshapes_matrix)

        for sdx, seq_id in enumerate(tqdm(seq_id_list, leave=False)):
            mesh_seq_list = dataset.get_mesh_seq(person_id, seq_id)
            if len(mesh_seq_list) == 0:
                continue

            mesh_seq_vertices_list = [mesh.vertices for mesh in mesh_seq_list]
            mesh_seq_vertices_vector_list = []
            if head_idx_list is None:
                mesh_seq_vertices_vector_list = [
                    vertices.reshape((-1, 1)) for vertices in mesh_seq_vertices_list
                ]
            else:
                mesh_seq_vertices_vector_list = [
                    vertices[head_idx_list].reshape((-1, 1))
                    for vertices in mesh_seq_vertices_list
                ]

            # Solve Optimization problem
            w_soln = opt_prob.optimize(mesh_seq_vertices_vector_list)

            # Save outputs
            out_path = coeff_out_path(person_id, seq_id, sdx > 0)
            np.save(out_path, w_soln)

if __name__ == "__main__":
    main()
