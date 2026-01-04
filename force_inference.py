"""
Force Inference for Epithelial Cells

Python implementation of Bayesian Force Inference technique by Ishihara and Sugimura
(Journal of Theoretical Biology, 2012).

Author: Converted from Mathematica by Claude
Original Author: Ali Hashmi

This script infers tension for edges between epithelial cells and the pressure within the cells.
The input is a binarized image of an epithelia.
"""

import numpy as np
from scipy import sparse, ndimage
from scipy.spatial import distance_matrix
from scipy.linalg import qr
from skimage import morphology, measure, segmentation, filters
from skimage.draw import line
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection, PatchCollection
import warnings
warnings.filterwarnings('ignore')


class ForceInference:
    """Main class for force inference analysis on epithelial cell images."""

    def __init__(self, image_path, has_border=True):
        """
        Initialize ForceInference object.

        Parameters:
        -----------
        image_path : str
            Path to the binarized image file
        has_border : bool
            Whether the image has filled borders (default: True)
        """
        self.image_path = image_path
        self.has_border = has_border
        self.image = None
        self.segmentation = None
        self.max_cell_labels = 0

    def segment_image(self, binarized_mask, border=True):
        """
        Segment the binarized mask into cell regions.

        Parameters:
        -----------
        binarized_mask : ndarray
            Binary image of cells
        border : bool
            Whether border is present

        Returns:
        --------
        ndarray : Labeled segmentation
        """
        if not border:
            print("Border not present, segmenting with morphological components")
            # Invert, find connected components, remove border components
            inverted = ~binarized_mask
            labeled = measure.label(inverted, connectivity=1)
            # Remove border components
            mask = np.ones_like(labeled, dtype=bool)
            mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = False
            for region in measure.regionprops(labeled):
                if np.any(~mask[region.coords[:, 0], region.coords[:, 1]]):
                    labeled[labeled == region.label] = 0
            return measure.label(labeled > 0, connectivity=1)
        else:
            print("Border exists, segmenting with watershed components")
            return segmentation.watershed(binarized_mask, connectivity=1)

    def bresenham_line(self, p0, p1):
        """
        Find intermediate pixels connecting two pixels using Bresenham's algorithm.

        Parameters:
        -----------
        p0, p1 : tuple
            Start and end points (y, x)

        Returns:
        --------
        list : List of points along the line
        """
        y0, x0 = int(p0[0]), int(p0[1])
        y1, x1 = int(p1[0]), int(p1[1])

        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            points.append((y, x))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points

    def close_image(self, image):
        """
        Artificially close cells if the surrounding/peripheral cells are open.

        Parameters:
        -----------
        image : ndarray
            Binary image

        Returns:
        --------
        ndarray : Closed image
        """
        # Find skeleton endpoints
        skeleton = morphology.skeletonize(image)
        # Find endpoints (pixels with only one neighbor in skeleton)
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        endpoints = np.where((neighbor_count == 11) & skeleton)

        if len(endpoints[0]) == 0:
            return image

        endpoint_coords = list(zip(endpoints[0], endpoints[1]))

        # Find center and sort by angle
        mean_pos = np.mean(endpoint_coords, axis=0)
        angles = [np.arctan2(p[0] - mean_pos[0], p[1] - mean_pos[1]) for p in endpoint_coords]
        sorted_indices = np.argsort(angles)
        pts_ordered = [endpoint_coords[i] for i in sorted_indices]
        pts_ordered.append(pts_ordered[0])  # Close the loop

        # Draw lines between consecutive points
        result = image.copy()
        for i in range(len(pts_ordered) - 1):
            line_points = self.bresenham_line(pts_ordered[i], pts_ordered[i + 1])
            for p in line_points:
                if 0 <= p[0] < image.shape[0] and 0 <= p[1] < image.shape[1]:
                    result[p[0], p[1]] = True

        return result

    def associate_vertices(self, img, seg, stringent_check=True):
        """
        Associate cells to their respective vertices.

        Parameters:
        -----------
        img : ndarray
            Binary skeleton image
        seg : ndarray
            Segmented image
        stringent_check : bool
            Whether to use stringent merging criteria

        Returns:
        --------
        dict : Mapping of cell combinations to vertex positions
        """
        # Find branch points (vertices where 3+ edges meet)
        skeleton = morphology.skeletonize(img)

        # Detect branch points using hit-or-miss transform
        # A branch point has 3 or more neighbors
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        neighbor_count = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        branch_points = np.where((neighbor_count >= 13) & skeleton)  # 10 + 3 neighbors

        pts = list(zip(branch_points[0], branch_points[1]))

        if len(pts) == 0:
            return {}

        # For each branch point, find neighboring cells
        members = []
        for pt in pts:
            y, x = pt
            # Create small dilation around point
            y_min, y_max = max(0, y-1), min(seg.shape[0], y+2)
            x_min, x_max = max(0, x-1), min(seg.shape[1], x+2)

            region = seg[y_min:y_max, x_min:x_max]
            unique_cells = np.unique(region)
            unique_cells = unique_cells[unique_cells > 0]  # Remove background
            members.append(sorted(unique_cells.tolist()))

        # Extract vertices with 2 or more neighboring cells
        vertices = {}
        for i, member_list in enumerate(members):
            if len(member_list) >= 2:
                key = tuple(sorted(member_list))
                if key not in vertices:
                    vertices[key] = []
                vertices[key].append(pts[i])

        # Merge nearby vertices (within 3 pixels)
        if stringent_check:
            merged_vertices = {}
            for key, positions in vertices.items():
                if len(positions) == 1:
                    merged_vertices[key] = positions[0]
                else:
                    # Check if positions are close together
                    positions_array = np.array(positions)
                    # If all positions are within 3 pixels, merge them
                    centroid = np.mean(positions_array, axis=0)
                    distances = np.linalg.norm(positions_array - centroid, axis=1)
                    if np.max(distances) < 3:
                        merged_vertices[key] = tuple(centroid)
                    else:
                        # Keep them separate
                        for i, pos in enumerate(positions):
                            merged_vertices[key + (i,)] = pos
            return merged_vertices
        else:
            # Simple merging: average positions for same cell combinations
            return {key: tuple(np.mean(positions, axis=0)) for key, positions in vertices.items()}

    def form_and_compute_matrices(self, vertex_coord_lookup, inds, cols_order, edge_num,
                                   del_v, vertex_to_cells, vertex_vertex_conn,
                                   max_cell_labels, filtered_vertices, vertex_assoc):
        """
        Generate the relevant matrices for force computation.

        Parameters:
        -----------
        vertex_coord_lookup : dict
            Mapping from vertex index to coordinates
        inds : list
            Edge indices as pairs of vertices
        cols_order : list
            Column ordering for edges
        edge_num : int
            Number of edges
        del_v : list
            Vertices to delete
        vertex_to_cells : dict
            Mapping from vertex to neighboring cells
        vertex_vertex_conn : list
            Vertex-vertex connections
        max_cell_labels : int
            Maximum cell label
        filtered_vertices : list
            List of valid vertices
        vertex_assoc : dict
            Vertex coordinate to index mapping

        Returns:
        --------
        tuple : (spArrayX, spArrayY, dimTx, dimPx)
        """
        # Compute tension coefficients (unit vectors along edges)
        tx, ty = [], []
        for source_idx, target_idx in inds:
            source = np.array(vertex_coord_lookup[source_idx])
            target = np.array(vertex_coord_lookup[target_idx])
            diff = target - source
            norm = np.linalg.norm(diff)
            if norm > 0:
                unit_vec = diff / norm
                ty.append(unit_vec[0])  # y component
                tx.append(unit_vec[1])  # x component
            else:
                ty.append(0)
                tx.append(0)

        print(f"Tension coefficients computed: ✓")
        print(f"Counts of zero coefficients Tx: {tx.count(0)}")
        print(f"Counts of zero coefficients Ty: {ty.count(0)}")

        # Filter vertices
        filtered_vertices_clean = [v for v in filtered_vertices if v not in del_v]
        filtered_vertex_num = len(filtered_vertices_clean)
        relabel_vert = {v: i for i, v in enumerate(filtered_vertices_clean)}

        # Build tension sparse matrices
        tens_inds = [(relabel_vert[inds[i][0]], cols_order[i]) for i in range(len(inds))
                     if inds[i][0] in relabel_vert]

        sp_array_tx = sparse.lil_matrix((filtered_vertex_num, edge_num))
        sp_array_ty = sparse.lil_matrix((filtered_vertex_num, edge_num))

        for idx, (i, j) in enumerate(tens_inds):
            sp_array_tx[i, j] = tx[idx]
            sp_array_ty[i, j] = ty[idx]

        sp_array_tx = sp_array_tx.tocsr()
        sp_array_ty = sp_array_ty.tocsr()

        # Build pressure matrices
        sp_array_px = sparse.lil_matrix((filtered_vertex_num, max_cell_labels))
        sp_array_py = sparse.lil_matrix((filtered_vertex_num, max_cell_labels))

        # Cell to vertex mapping
        cell_to_vertex_labels = {}
        for vertex_idx, cells in vertex_to_cells.items():
            for cell in cells:
                if cell not in cell_to_vertex_labels:
                    cell_to_vertex_labels[cell] = []
                cell_to_vertex_labels[cell].append(vertex_idx)

        # Process each vertex for pressure coefficients
        kk = 0
        for i in filtered_vertices_clean:
            if i not in vertex_to_cells:
                continue

            neighboring_cells = vertex_to_cells[i]

            if len(neighboring_cells) > 2:
                # Get vertices for each neighboring cell
                vertex_coords_list = []
                cell_list = []

                for cell in neighboring_cells:
                    if cell in cell_to_vertex_labels:
                        cell_vertices = cell_to_vertex_labels[cell]
                        # Find vertices adjacent to current vertex i
                        adjacent = [v for v in cell_vertices if v != i and v in vertex_coord_lookup]
                        if len(adjacent) >= 1:
                            vertex_coords_list.append(vertex_coord_lookup[adjacent[0]])
                            cell_list.append(cell)

                if len(vertex_coords_list) >= 2:
                    # Sort by angle around centroid
                    vertex_coords = np.array(vertex_coords_list)
                    centroid = np.mean(vertex_coords, axis=0)
                    angles = np.arctan2(vertex_coords[:, 0] - centroid[0],
                                       vertex_coords[:, 1] - centroid[1])
                    order = np.argsort(angles)

                    ordered_cells = [cell_list[j] for j in order]
                    ordered_coords = vertex_coords[order]

                    # Compute pressure coefficients (perpendicular to edges)
                    for j in range(len(ordered_coords)):
                        next_j = (j + 1) % len(ordered_coords)
                        edge_vec = ordered_coords[next_j] - ordered_coords[j]
                        # Perpendicular vector (rotated 90 degrees)
                        px = edge_vec[0] / 2
                        py = -edge_vec[1] / 2

                        if px == 0 and py == 0:
                            kk += 1

                        if i in relabel_vert and ordered_cells[j] <= max_cell_labels:
                            row_idx = relabel_vert[i]
                            col_idx = ordered_cells[j] - 1  # 0-indexed
                            if col_idx < max_cell_labels:
                                sp_array_px[row_idx, col_idx] = px
                                sp_array_py[row_idx, col_idx] = py

        sp_array_px = sp_array_px.tocsr()
        sp_array_py = sp_array_py.tocsr()

        print(f"Pressure coefficients computed: ✓")
        print(f"Pressure coefficients zero: {kk}")

        # Combine tension and pressure matrices
        sp_array_x = sparse.hstack([sp_array_tx, sp_array_px])
        sp_array_y = sparse.hstack([sp_array_ty, sp_array_py])

        dim_tx = sp_array_tx.shape
        dim_px = sp_array_px.shape

        return sp_array_x, sp_array_y, dim_tx, dim_px

    def maximize_log_likelihood(self, sp_array_x, sp_array_y, dim_tx, dim_px):
        """
        Maximize log-likelihood function to find optimal tension and pressure values.

        Parameters:
        -----------
        sp_array_x, sp_array_y : sparse matrices
            Combined tension and pressure coefficient matrices
        dim_tx, dim_px : tuple
            Dimensions of tension and pressure matrices

        Returns:
        --------
        ndarray : Optimized parameters (tensions and pressures)
        """
        print("\n=== Maximum Likelihood Optimization ===")

        # Parameter range for mu
        mu_range = 10.0 ** np.arange(-1.5, 1.6, 0.1)

        # Stack matrices
        sp_a = sparse.vstack([sp_array_x, sp_array_y])

        # Gamma vector: 1 for tensions, 0 for pressures
        gamma = np.concatenate([np.ones(dim_tx[1]), np.zeros(dim_px[1])])
        sp_gamma = sparse.csr_matrix(gamma).T
        sp_b_mat = sparse.diags(gamma)

        n = sp_a.shape[0]
        m = dim_px[1]  # Number of pressure parameters

        sp_b_vec = sparse.csr_matrix((sp_a.shape[0], 1))

        log_likelihoods = []

        for mu in mu_range:
            tau = np.sqrt(mu)

            # Form augmented matrix [A; tau*B] and vector [b; tau*gamma]
            s_matrix = sparse.vstack([sp_a, tau * sp_b_mat])
            s_vector = sparse.vstack([sp_b_vec, tau * sp_gamma])

            # Combine into single matrix for QR decomposition
            s_combined = sparse.hstack([s_matrix, s_vector]).toarray()

            # QR decomposition
            Q, R = qr(s_combined, mode='economic')

            # Adjust R for sign consistency
            signs = np.sign(np.diag(R))
            R = np.diag(signs) @ R

            num_params = sp_b_mat.shape[0]
            H = R[:num_params, :num_params]
            h_vec = R[:num_params, num_params]
            h_scalar = R[num_params, num_params] if R.shape[0] > num_params else 1e-10

            # Compute log-likelihood
            try:
                term1 = -(n - m + 1) * np.log(h_scalar**2 + 1e-10)

                # Diagonal of mu * B^T * B for non-zero gamma
                diag_vals = mu * gamma[gamma > 0]
                term2 = np.sum(np.log(diag_vals + 1e-10))

                # Diagonal of H (excluding last row/col)
                h_diag = np.diag(H[:-1, :-1]) if H.shape[0] > 1 else np.array([H[0, 0]])
                term3 = -2 * np.sum(np.log(np.abs(h_diag) + 1e-10))

                log_l = term1 + term2 + term3
                log_likelihoods.append(log_l)
            except:
                log_likelihoods.append(-np.inf)

        # Plot log-likelihood
        plt.figure(figsize=(8, 5))
        plt.plot(log_likelihoods, 'r-', linewidth=2)
        plt.plot(log_likelihoods, 'ko', markersize=4)
        plt.xlabel('Index μ')
        plt.ylabel('Log-Likelihood')
        plt.title('Log-Likelihood vs μ')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('log_likelihood.png', dpi=150, bbox_inches='tight')
        print(f"Log-likelihood plot saved to log_likelihood.png")

        # Find optimal mu
        max_idx = np.argmax(log_likelihoods)
        mu_opt = mu_range[max_idx]
        print(f"Optimized value of μ: {mu_opt:.6f}")

        # Compute final solution with optimal mu
        tau_opt = np.sqrt(mu_opt)
        s_matrix = sparse.vstack([sp_a, tau_opt * sp_b_mat])
        s_vector = sparse.vstack([sp_b_vec, tau_opt * sp_gamma])
        s_combined = sparse.hstack([s_matrix, s_vector]).toarray()

        Q, R = qr(s_combined, mode='economic')
        signs = np.sign(np.diag(R))
        R = np.diag(signs) @ R

        num_params = sp_b_mat.shape[0]
        H = R[:num_params, :num_params]
        h_vec = R[:num_params, num_params]

        # Solve H * p = h_vec using pseudo-inverse
        try:
            p = np.linalg.lstsq(H, h_vec, rcond=None)[0]
        except:
            p = np.zeros(num_params)

        return p

    def plot_maps(self, p, seg, edge_img, max_cell_labels, dim_tx,
                  vertex_to_cells, vertex_coord_lookup, edge_labels, border):
        """
        Generate plots for tension and pressure maps.

        Parameters:
        -----------
        p : ndarray
            Solution vector (tensions and pressures)
        seg : ndarray
            Segmented image
        edge_img : list
            Edge line segments
        max_cell_labels : int
            Maximum number of cells
        dim_tx : tuple
            Dimensions of tension matrix
        vertex_to_cells : dict
            Vertex to cells mapping
        vertex_coord_lookup : dict
            Vertex coordinates
        edge_labels : dict
            Edge to vertex pairs
        border : bool
            Whether image has border
        """
        # Cell to vertices mapping
        cell_to_vertex_labels = {}
        for vertex_idx, cells in vertex_to_cells.items():
            for cell in cells:
                if cell not in cell_to_vertex_labels:
                    cell_to_vertex_labels[cell] = []
                cell_to_vertex_labels[cell].append(vertex_idx)

        # Extract tension values
        t_vals = p[:dim_tx[1]]
        t_vals_normalized = (t_vals - np.min(t_vals)) / (np.max(t_vals) - np.min(t_vals) + 1e-10)

        # Plot tension map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Tension map
        cmap = plt.cm.rainbow
        colors = [cmap(val) for val in t_vals_normalized]

        line_segments = []
        line_colors = []
        for i, (edge_idx, (v1, v2)) in enumerate(edge_labels.items()):
            if v1 in vertex_coord_lookup and v2 in vertex_coord_lookup:
                coord1 = vertex_coord_lookup[v1]
                coord2 = vertex_coord_lookup[v2]
                line_segments.append([coord1[::-1], coord2[::-1]])  # Reverse for plotting
                if i < len(colors):
                    line_colors.append(colors[i])

        lc = LineCollection(line_segments, colors=line_colors, linewidths=2)
        ax1.add_collection(lc)
        ax1.set_xlim(0, seg.shape[1])
        ax1.set_ylim(seg.shape[0], 0)
        ax1.set_aspect('equal')
        ax1.set_title('Tension Map', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Pressure map
        p_vals = p[dim_tx[1]:]

        # Find border cells to exclude
        if border:
            border_cells = set()
            # Cells touching image border
            border_mask = np.zeros_like(seg, dtype=bool)
            border_mask[0, :] = border_mask[-1, :] = border_mask[:, 0] = border_mask[:, -1] = True
            for region in measure.regionprops(seg):
                if np.any(border_mask[region.coords[:, 0], region.coords[:, 1]]):
                    border_cells.add(region.label)
            remove_cells = list(border_cells)
        else:
            remove_cells = []

        cell_labels = [i for i in range(1, max_cell_labels + 1) if i not in remove_cells]

        # Create polygons for cells
        polygons = []
        p_colors = []

        for cell_label in cell_labels:
            if cell_label not in cell_to_vertex_labels:
                continue

            vertices = cell_to_vertex_labels[cell_label]
            if len(vertices) < 3:
                continue

            # Get vertex coordinates
            pts = [vertex_coord_lookup[v] for v in vertices if v in vertex_coord_lookup]
            if len(pts) < 3:
                continue

            # Sort by angle around centroid
            pts_array = np.array(pts)
            centroid = np.mean(pts_array, axis=0)
            angles = np.arctan2(pts_array[:, 0] - centroid[0], pts_array[:, 1] - centroid[1])
            order = np.argsort(angles)
            ordered_pts = pts_array[order]

            # Reverse coordinates for plotting (x, y instead of y, x)
            poly_coords = ordered_pts[:, ::-1]
            polygons.append(MplPolygon(poly_coords, True))

            # Get pressure value
            if cell_label - 1 < len(p_vals):
                p_colors.append(p_vals[cell_label - 1])
            else:
                p_colors.append(0)

        if polygons:
            p_colors_normalized = (np.array(p_colors) - np.min(p_colors)) / (np.max(p_colors) - np.min(p_colors) + 1e-10)
            patch_colors = [cmap(val) for val in p_colors_normalized]

            pc = PatchCollection(polygons, facecolors=patch_colors, edgecolors='black', linewidths=0.5)
            ax2.add_collection(pc)

        # Add edges to pressure map
        lc2 = LineCollection(line_segments, colors='black', linewidths=1, alpha=0.5)
        ax2.add_collection(lc2)

        ax2.set_xlim(0, seg.shape[1])
        ax2.set_ylim(seg.shape[0], 0)
        ax2.set_aspect('equal')
        ax2.set_title('Pressure Map', fontsize=14, fontweight='bold')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig('force_inference_results.png', dpi=150, bbox_inches='tight')
        print(f"\nResults saved to force_inference_results.png")
        plt.show()

    def run(self):
        """
        Main function to run force inference analysis.
        """
        print("="*60)
        print("FORCE INFERENCE FOR EPITHELIAL CELLS")
        print("="*60)

        # Load and binarize image
        img_pil = Image.open(self.image_path).convert('L')
        img_array = np.array(img_pil)

        # Binarize
        threshold = filters.threshold_otsu(img_array)
        self.image = img_array > threshold

        # Close image if no border
        if not self.has_border:
            self.image = self.close_image(self.image)

        print(f"\nImage loaded: {self.image.shape}")

        # Segment image
        self.segmentation = self.segment_image(self.image, self.has_border)
        self.max_cell_labels = np.max(self.segmentation)
        print(f"Image segmented: ✓")
        print(f"Number of cells: {self.max_cell_labels}")

        # Associate vertices
        cells_to_vertices = self.associate_vertices(self.image, self.segmentation)
        print(f"Vertices found and associated: ✓")
        print(f"Number of vertices: {len(cells_to_vertices)}")

        # Find edges
        skeleton = morphology.skeletonize(self.image)
        edges = measure.label(skeleton, connectivity=2)

        print(f"Edges found and associated: ✓")

        # Create vertex index mapping
        all_vertices = list(cells_to_vertices.values())
        vertex_assoc = {tuple(v): i for i, v in enumerate(all_vertices)}
        vertex_coord_lookup = {i: tuple(v) for i, v in enumerate(all_vertices)}

        # Create vertex to cells mapping
        vertex_to_cells = {}
        for cells, vertex_pos in cells_to_vertices.items():
            vertex_idx = vertex_assoc[tuple(vertex_pos)]
            vertex_to_cells[vertex_idx] = list(cells)

        # Filter vertices with 3+ cells
        filtered_vertices = [v_idx for v_idx, cells in vertex_to_cells.items() if len(cells) > 2]
        print(f"Filtered vertices (3+ cells): {len(filtered_vertices)}")

        # Build edge list (simplified version)
        edge_labels = {}
        edge_idx = 0
        vertex_list = list(vertex_coord_lookup.keys())

        for i, v1 in enumerate(vertex_list):
            for v2 in vertex_list[i+1:]:
                # Check if vertices share cells
                if v1 in vertex_to_cells and v2 in vertex_to_cells:
                    common_cells = set(vertex_to_cells[v1]) & set(vertex_to_cells[v2])
                    if len(common_cells) >= 1:
                        edge_labels[edge_idx] = (v1, v2)
                        edge_idx += 1

        edge_num = len(edge_labels)
        print(f"Number of edges: {edge_num}")

        # Prepare indices for matrix formation
        inds = [(v1, v2) for v1, v2 in edge_labels.values()]
        cols_order = list(range(edge_num))

        # Build vertex-vertex connections
        vertex_vertex_conn = {v: [] for v in filtered_vertices}
        for v1, v2 in edge_labels.values():
            if v1 in filtered_vertices:
                vertex_vertex_conn[v1].append((v1, v2))
            if v2 in filtered_vertices:
                vertex_vertex_conn[v2].append((v2, v1))

        del_v = []  # Vertices to delete (if any have issues)

        # Form and compute matrices
        sp_array_x, sp_array_y, dim_tx, dim_px = self.form_and_compute_matrices(
            vertex_coord_lookup, inds, cols_order, edge_num, del_v,
            vertex_to_cells, vertex_vertex_conn, self.max_cell_labels,
            filtered_vertices, vertex_assoc
        )

        # Maximize log-likelihood
        p = self.maximize_log_likelihood(sp_array_x, sp_array_y, dim_tx, dim_px)

        # Plot results
        self.plot_maps(p, self.segmentation, edge_labels, self.max_cell_labels,
                      dim_tx, vertex_to_cells, vertex_coord_lookup, edge_labels,
                      self.has_border)

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        return p


def main():
    """Example usage of ForceInference class."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python force_inference.py <image_path> [has_border]")
        print("\nExample:")
        print("  python force_inference.py image.tif True")
        print("  python force_inference.py image.tif False")
        return

    image_path = sys.argv[1]
    has_border = True if len(sys.argv) < 3 else sys.argv[2].lower() == 'true'

    # Run force inference
    fi = ForceInference(image_path, has_border=has_border)
    result = fi.run()

    print(f"\nComputed {len(result)} parameters (tensions + pressures)")


if __name__ == "__main__":
    main()
