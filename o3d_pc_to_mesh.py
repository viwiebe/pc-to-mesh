import sys
import argparse
import subprocess
import laspy
import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import imageio.v3 as iio
import tifffile
from PIL import Image
import OpenEXR
import Imath
import tetgen


pc_name = "pc"
output_path = "results"
original_pc = None
used_pc = None
debug = False
visualize = False

# default values
voxel_size_value = 0.2

nb_neighbors_value = 50
std_ratio_value = 2.0

nb_points_value = 6.0
radius_value = 1.0

# poisson values
octree_depth = 9
poisson_density_cutoff = 0.01

# las header values
z_min = 0
z_max = 1


def main():
    global pc_name, output_path, original_pc, used_pc, debug, visualize
    global voxel_size_value, nb_neighbors_value, std_ratio_value, nb_points_value, radius_value, use_tetgen, octree_depth
    global z_min, z_max

    # usage: python SCRIPTNAME [--options] "PATH/TO/CLOUD" "PATH/TO/RESULTS" 
    parser = argparse.ArgumentParser(description="Clean a point cloud for surface reconstruction as mesh for Godot.")
    parser.add_argument("input", help="Path to input point cloud")
    parser.add_argument("outpath", help="Path to output results")
    parser.add_argument("--dimensions", action="store_true", help="Print dimension names to find normals")
    parser.add_argument("--name", default="cleaned_pc", help="Output image filename, default = cleaned_pc")
    # possible arguments for this: all or multiple of: downsample, outliers, mesh (= alpha), poisson, classification, densify, ball
    parser.add_argument("--parts", nargs='+', default="all", help="Which functions should be executed (default=all, options=downsample, outliers, alpha, poisson, ball)")
    parser.add_argument("--outliers", nargs=4, help="Values for nb neighbors, std ratio, nb points and radius in that order")
    parser.add_argument("--alpha", nargs='+', default="", help="alpha shape alpha values as a list")
    parser.add_argument("--octree", help="Octree depth for Poisson Recon")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--visualize", action="store_true", help="Executes visualization after processing")
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.debug:
        debug = True

    if args.visualize:
        visualize = True

    las = laspy.read(args.input)
    if las is None:
        print("Error loading image.")
        return
    
    if args.outpath:
        output_path = args.outpath

    # save las header values
    z_min = las.header.mins[2]
    z_max = las.header.maxs[2]

    if debug:
        print("Header min and max...")
        print(f"Min: {z_min} | Max: {z_max}")
    
    if args.dimensions:
        print("Dimensions...")
        print(list(las.point_format.dimension_names))
        return

    if args.name:
        pc_name = args.name
        if debug:
            print(f"PC Name: {pc_name}")

    parts = args.parts

    if args.outliers is not None:
        nb, std, voxel, radius = args.outliers
        nb_neighbors_value = int(nb)
        std_ratio_value = float(std)
        voxel_size_value = float(voxel)
        radius_value = float(radius)

    if "classification" in parts:
        mesh_from_classifications(las)
        # TODO: if no other options: terminate here

    used_pc = to_open3d_cloud(las)
    original_pc = to_open3d_cloud(las)

    if "all" in parts or "downsample" in parts:
        used_pc = downsample_cloud(used_pc)

    if "all" in parts or "outliers" in parts:
        used_pc, ind = remove_outliers(used_pc)

    if "all" in parts or "alpha" in parts:
        alpha_shape(used_pc)

    if "all" in parts or "poisson" in parts:
        if args.octree:
            octree_depth = int(args.octree)
        poisson_recon(pcd=used_pc)

    if "all" in parts or "heightmap" in parts:
        generate_heightmap(pcd=used_pc, debug_run=False)
        if debug:
            generate_heightmap(pcd=original_pc, debug_run=True)

    if "all" in parts or "convex" in parts:
        pc_to_mesh(pcd=used_pc)
    
    if "all" in parts or "delaunay" in parts:
        delaunay(pcd=used_pc)

    if "all" in parts or "ball" in parts:
        ball_pivoting(pcd=used_pc)


def print_help():
    print("How to use Open3D Point Cloud to Mesh:")
    print("")


def to_open3d_cloud(las):
    global pc_name, debug

    print("Converting las cloud to open3d cloud...")
    points = np.vstack((las.x, las.y, las.z)).transpose()

    try:
        # add your own normal keyword here if it differs from the ones used here
        print(las)
        if "NormalX" in las:
            normals = np.vstack((
                las["NormalX"],
                las["NormalY"],
                las["NormalZ"]
            )).transpose()
        else:
            normals = np.vstack((
                las["X"],
                las["Y"],
                las["Z"]
            )).transpose()
    except (KeyError, ValueError) as e:
        print("No normals found: %s", e)

    # Normalize the normals if not already unit vectors
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    # Optionally, assign colors if available (e.g., RGB)
    if hasattr(las, 'red'):
        rgb = np.vstack((las.red, las.green, las.blue)).transpose()
        rgb = (rgb / 65535.0).astype(np.float64)  # Normalize to [0, 1] if 16-bit
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Save to .ply or .xyz -> TODO: change to obj
    if debug:
        o3d.io.write_point_cloud(f"results/{pc_name}_converted.ply", pcd)
    
    return pcd


def downsample_cloud(pcd):
    # TODO downsample and outlier removal does nothing
    global pc_name, voxel_size_value, debug

    print("Downsampling cloud...")
    print(f"Downsample the point cloud with a voxel of {voxel_size_value}")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size_value)
    o3d.io.write_point_cloud(f"{pc_name}_voxeldown.ply", voxel_down_pcd)
    return voxel_down_pcd


def remove_outliers(pcd):
    # TODO give option to do both? Or just do both?
    global nb_points_value, radius_value
    print("Remove statistical ouliers...")
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors_value, std_ratio=std_ratio_value)
    cl, ind = pcd.remove_radius_outlier(nb_points=6, radius=1.0)
    # o3d.io.write_point_cloud("converted_voxeldown_outlier.ply", cl)
    return cl, ind


def alpha_shape(pcd):
    global visualize

    print("Create alpha shape mesh...")

    # Center and scale to unit-ish size
    center = pcd.get_center()
    pcd.translate(-center)

    bounds = pcd.get_max_bound() - pcd.get_min_bound()
    scale = 1.0 / bounds.max()  # scale so largest dimension = 1
    pcd.scale(scale, center=(0,0,0))

    nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    d = np.median(nn_dists)
    print(f"Median neighbor distance: {d}")

    alphas = [2 * d, 3 * d, 5 * d, 8 * d, 10 * d]

    for alpha in alphas:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        if visualize:
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
        
        o3d.io.write_triangle_mesh(
            f"{pc_name}_{alpha}_alpha.obj",
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True
        )


def ball_pivoting(pcd):
    global pc_name
    
    print("Create ball pivoting...")
    factor_list=[1.5, 2.5, 5, 10]
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    # estimate typical spacing (nearest neighbor distance)
    nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    # robust measure: median (less sensitive to outliers)
    d = float(np.median(nn_dists))
    if d == 0 or np.isnan(d):
        d = float(np.mean(nn_dists[np.isfinite(nn_dists)]))

    print(f"Estimated median nearest-neighbor distance d = {d:.6f}")

    # build radii from factors
    radii = [d * f for f in factor_list]
    print("Trying radii:", radii)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, 
        o3d.utility.DoubleVector(radii)
    )

    center = mesh.get_center()
    mesh.translate(-center)

    scale_and_clean(mesh)
    
    if visualize:
        o3d.visualization.draw_geometries([pcd, rec_mesh])
    
    # create obj mesh
    o3d.io.write_triangle_mesh(
        f"{pc_name}_meshbpa.obj",
        mesh,
        write_vertex_normals=True,
        write_vertex_colors=True
    )


def poisson_recon(pcd):
    global octree_depth, poisson_density_cutoff, pc_name, output_path

    print("Run Poisson surface reconstruction...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=octree_depth)
        vertices_to_remove = densities < np.quantile(densities, poisson_density_cutoff)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        center = mesh.get_center()
        mesh.translate(-center)

        scale_and_clean(mesh)

        o3d.io.write_triangle_mesh(
            f"{output_path}/{pc_name}_poisson.obj",
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True
        )
        print("Reconstruction complete")
        return mesh


def scale_and_clean(mesh):
    
    # Scale mesh to 10 units in largest dimension
    bounds = mesh.get_max_bound() - mesh.get_min_bound()
    scale_factor = 10.0 / max(bounds)  # scale largest dimension to 10
    mesh.scale(scale_factor, center=np.array([0, 0, 0]))

    # Rotate mesh from Z-up (Open3D) to Y-up (Godot)
    R = mesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))  # rotate -90° around X
    mesh.rotate(R, center=np.array([0, 0, 0]))

    # Clean up the mesh
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles() 
    mesh.remove_non_manifold_edges() 
    mesh.remove_unreferenced_vertices() 

    # Recompute valid normals
    mesh.compute_vertex_normals()


def generate_heightmap(pcd, debug_run):
    global pc_name, debug, z_min, z_max

    print("Generating heightmap...")
    points = np.asarray(pcd.points)

    z_range = z_max - z_min

    # Set grid resolution (size of each cell)
    grid_size = 1.0  # meters

    # Compute bounds
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)

    # Define grid dimensions
    nx = int(np.ceil((max_x - min_x) / grid_size))
    ny = int(np.ceil((max_y - min_y) / grid_size))

    # Initialize heightmap with NaNs
    heightmap = np.full((ny, nx), np.nan)

    # Populate with max height (Z) in each cell 
    for x, y, z in points:
        ix = int((x - min_x) / grid_size)
        iy = int((y - min_y) / grid_size)
        if 0 <= ix < nx and 0 <= iy < ny:
            if np.isnan(heightmap[iy, ix]) or z > heightmap[iy, ix]:
                heightmap[iy, ix] = z

    # Interpolate missing values using scipy
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))
    valid_mask = ~np.isnan(heightmap)
    known_x = xx[valid_mask]
    known_y = yy[valid_mask]
    known_z = heightmap[valid_mask]

    # Interpolation method: 'linear', 'nearest', or 'cubic'
    filled_heightmap = griddata(
        (known_x, known_y),
        known_z,
        (xx, yy),
        method='cubic'
    )

    # Optional: fallback to nearest for remaining NaNs
    nan_mask = np.isnan(filled_heightmap)
    if np.any(nan_mask):
        filled_heightmap[nan_mask] = griddata(
            (known_x, known_y),
            known_z,
            (xx[nan_mask], yy[nan_mask]),
            method='nearest'
        )
    
    filled_heightmap -= np.nanmin(filled_heightmap)

    # convert to float32 (required for EXR)
    filled_heightmap_float = filled_heightmap.astype(np.float32)
    heightmap_tmp = np.clip(filled_heightmap_float, 0, 1)
    heightmap_tmp_normalized = filled_heightmap.astype(np.float32)

    normalized = (filled_heightmap_float - z_min) / z_range

    # Encode into 24-bit RGB
    h_int = (normalized * 16777215).astype(np.uint32)
    R = ((h_int >> 16) & 0xFF).astype(np.uint8)
    G = ((h_int >> 8) & 0xFF).astype(np.uint8)
    B = (h_int & 0xFF).astype(np.uint8)

    # Stack into (H, W, 3) array
    rgb_image = np.stack([R, G, B], axis=-1)

    # Convert to float32 for EXR output, scale to [0,1]
    rgb_image_f32 = rgb_image.astype(np.float32) / 255.0

    # Save directly to EXR with RGB channels
    exr_path = f"results/{pc_name}_heightmap.exr"
    save_rgb_exr(exr_path, R, G, B)
    # also save as png image
    Image.fromarray(rgb_image, mode='RGB').save(f"results/{pc_name}_heightmap_rgb_debug.png")

    print("... generated.")


def save_rgb_exr(path, R, G, B):
    height, width = R.shape

    # Define header
    header = OpenEXR.Header(width, height)
    half_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = {
        'R': half_chan,
        'G': half_chan,
        'B': half_chan
    }

    # Convert to float32 and flatten row-major to strings
    Rf = R.astype(np.float32).tobytes()
    Gf = G.astype(np.float32).tobytes()
    Bf = B.astype(np.float32).tobytes()

    # Write EXR
    exr = OpenEXR.OutputFile(path, header)
    exr.writePixels({'R': Rf, 'G': Gf, 'B': Bf})
    exr.close()


def pc_to_mesh(pcd):
    mesh, _ = o3d.geometry.PointCloud.compute_convex_hull(pcd)
    if viszalize:
        o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(
            f"{output_path}/{pc_name}_convex.obj",
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True
        )
    # o3d.io.write_triangle_mesh(f"results/convex_hull_{pc_name}.ply", mesh)


def delaunay(pcd):
    points = np.asarray(pcd.points)

    # project into XY plane
    xy = points[:, :2]
    z = points[:, 2]

    triangulation = Delaunay(xy)
    triangles = triangulation.simplices
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()

    if visualize:
        o3d.visualization.draw_geometries([mesh])
    
    o3d.io.write_triangle_mesh(
            f"{output_path}/{pc_name}_delaunay.obj",
            mesh,
            write_vertex_normals=True,
            write_vertex_colors=True
        )


def mesh_from_classifications(las):
    global debug, visualize, output_path, pc_name

    if debug:
        print("Unique classes:", set(las.classification))
    
    print("Creating mesh from las classifications")
    points = np.vstack((las.x, las.y, las.z)).T
    classes = las.classification

    ground_pts = points[classes == 2]
    building_pts = points[classes == 20] # usually 6 but in our data there is only 2, 20 and 24, 24 being all vegetation and buildings
    
    if debug:
        print("Building points (24): %s" , len(building_pts))
        print("Points 20: %s", len(points[classes == 20]))

    # create ground mesh
    mesh = None

    # TODO: give option to choose which method to use instead of delaunay
    if len(ground_pts) > 3:
        tri = Delaunay(ground_pts[:, :2])
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(ground_pts)
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
        mesh.compute_vertex_normals()

    # building bounding boxes
    buildings = o3d.geometry.PointCloud()
    buildings.points = o3d.utility.Vector3dVector(building_pts)

    labels = np.array(buildings.cluster_dbscan(eps=1.5, min_points=30))
    boxes = []
    for cluster_id in np.unique(labels):
        if cluster_id == -1:
            continue
        cluster = buildings.select_by_index(np.where(labels == cluster_id)[0])
        box = cluster.get_axis_aligned_bounding_box()
        box.color = (1, 0, 0)
        boxes.append(box)
    
    box_meshes = []
    for box in boxes:
        box_mesh = bbox_to_mesh(box, color=(1,0,0))
        box_meshes.append(box_mesh)

    # visualize
    geoms = []
    if mesh: geoms.append(mesh)
    geoms += box_meshes

    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(ground_pts)
    pcd_ground.paint_uniform_color([0,1,0])

    pcd_buildings = o3d.geometry.PointCloud()
    pcd_buildings.points = o3d.utility.Vector3dVector(building_pts)
    pcd_buildings.paint_uniform_color([0,0,1])

    geoms += [pcd_ground, pcd_buildings]

    if visualize:
        o3d.visualization.draw_geometries(geoms)

    # save as obj
    combined_mesh = o3d.geometry.TriangleMesh()

    if mesh:
        combined_mesh += mesh
    for box_mesh in box_meshes:
        combined_mesh += box_mesh

    # Center mesh around origin and scale to manageable size
    center = combined_mesh.get_center()
    combined_mesh.translate(-center)

    bounds = combined_mesh.get_max_bound() - combined_mesh.get_min_bound()
    max_dim = bounds.max()
    if max_dim > 0:
        scale = 10.0 / max_dim  # scale so largest dimension is about 10 units
        combined_mesh.scale(scale, center=(0, 0, 0))
    
    scale_and_clean(combined_mesh)

    o3d.io.write_triangle_mesh(
        f"{output_path}/{pc_name}_classification_mesh.obj",
        combined_mesh,
        write_vertex_normals=True,
        write_vertex_colors=True
    )


def bbox_to_mesh(box, color=(1,0,0)):
    extent = box.get_extent()
    mesh = o3d.geometry.TriangleMesh.create_box(*extent)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    mesh.translate(box.get_min_bound())
    return mesh


if __name__ == "__main__":
    main()