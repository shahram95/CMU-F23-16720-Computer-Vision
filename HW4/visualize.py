import numpy as np
import matplotlib.pyplot as plt
import os.path
import submission as sub
import findM2
import plotly.graph_objs as go
import helper

def find_correspondences(image1, image2, fundamental_matrix, coordinates):
    num_coords = coordinates.shape[0]
    correspondences = np.zeros((num_coords, 2))
    for i in range(num_coords):
        x1, y1 = coordinates[i, 0], coordinates[i, 1]
        x2, y2 = sub.epipolarCorrespondence(image1, image2, fundamental_matrix, x1, y1)
        correspondences[i] = [x2, y2]
    return correspondences


def plot_3d_points(points_3d):
    scatter = go.Scatter3d(
        x=points_3d[:, 0], y=points_3d[:, 1], z=points_3d[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=points_3d[:, 2],  # set color to the Z coordinate for a depth effect
            colorscale='Viridis',  # choose a colorscale
            opacity=0.8
        )
    )

    fig = go.Figure(data=[scatter])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z', range=[3.4, 4.2])  # set the range for the z-axis
        ),
        width=700,
        height=700,
        title="3D Scatter Plot"
    )
    fig.show()

if __name__ == '__main__':
    # Load data
    correspondences_data = np.load('../data/some_corresp.npz')
    intrinsics_data = np.load('../data/intrinsics.npz')
    temple_coords_data = np.load('../data/templeCoords.npz')
    
    # Load images
    image1 = plt.imread('../data/im1.png')
    image2 = plt.imread('../data/im2.png')
    
    # Extract data
    K1, K2 = intrinsics_data['K1'], intrinsics_data['K2']
    points1 = correspondences_data['pts1']
    points2 = correspondences_data['pts2']
    x1, y1 = temple_coords_data['x1'], temple_coords_data['y1']
    
    # Compute fundamental matrix
    M = max(image1.shape)
    F = sub.eightpoint(points1.astype(np.float64), points2.astype(np.float64), M)
    
    # Find second camera matrix
    M1, C1, M2, C2, F = findM2.findM2(points1, points2, F, K1, K2)
    
    # Epipolar correspondences for the temple coordinates
    temple_points1 = np.hstack((x1, y1))
    temple_points2 = find_correspondences(image1, image2, F, temple_points1)
    
    # Triangulate points
    points_3d, error = sub.triangulate(C1, temple_points1, C2, temple_points2)
    
    # Plot the 3D points
    plot_3d_points(points_3d)
    

    np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)
