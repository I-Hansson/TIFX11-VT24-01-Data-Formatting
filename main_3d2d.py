import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import numpy as np
import os
import ffmpeg
import subprocess
#from camera import project_point_radial
CAMERA_DISC = {}

def project_onto_xz_plane(df):
    df_merged_points = pd.DataFrame(columns=['Name', 'Projected Points'])

    for index, row in df.iterrows():
        name = row['Name']
        points_3d = row['Values']


        projected_points = []

        for point in points_3d:
            x, y, z = point
            projected_points.append([x, z])  # Projecting onto x-z plane by discarding y-coordinate
        
        df_merged_points = df_merged_points._append({'Name': name, 'Projected Points': projected_points}, ignore_index=True)

    return df_merged_points


def project_2d(df):

    for index, row in df.iterrows():
            filename = row['File']
            coords = row['Values']
            current_name = filename.split('_')[0]
            #print(current_name)
            r_vec, _ = cv2.Rodrigues(CAMERA_DISC[filename]['camera_rotation'])  # Convers matrix into vector
            if len(coords) == 0:
                continue
            points_2d, _ = cv2.projectPoints(np.array(coords, dtype=np.float32), CAMERA_DISC[filename]['camera_rotation'], CAMERA_DISC[filename]['translation_vector'], CAMERA_DISC[filename]['camera_matrix'], CAMERA_DISC[filename]['dist_coeffs']) 
            #points_2d, _ = cv2.projectPoints(np.array(coords, dtype=np.float32), CAMERA_DISC[filename]['camera_rotation'], CAMERA_DISC[filename]['translation_vector'], CAMERA_DISC[filename]['focal_length'], CAMERA_DISC[filename]['center_point'], np.array(CAMERA_DISC[filename]['dist_coeffs'])) 
            #U_dist = points_2d[:, 0].T  
            #xn_u = cv2.undistortPoints(U_dist.T, CAMERA_DISC[filename]['camera_matrix'], CAMERA_DISC[filename]['dist_coeffs'], None, None)[:, 0].T  # Undistorts the projection 

            points_2d_flat = points_2d.reshape(-1, 2) 
            #print(points_2d_flat)
            # Rescales and moves the coordinates the coordinates to ensure that the origin is the right place. 
            x_coordinates = 2.2*points_2d_flat[:, 0] - CAMERA_DISC[filename]['origin'][current_name][0]
            y_coordinates = -2.2*points_2d_flat[:, 1] + CAMERA_DISC[filename]['origin'][current_name][1]
            x_list = x_coordinates.tolist()
            y_list = y_coordinates.tolist()

            
            #points_2d_list = [[int(p[0][0]), int(p[0][1])] for p in points_2d]

            # Create a list of [x, y] pairs
            
            coordinates_list = []
            for x, y in zip(x_list, y_list):
                coordinates_list.append([x, y])
            
            #df.loc[index, 'Values'] = points_2d_list
            df.loc[index, 'Values'] = coordinates_list
    return df
    




def plot_single_frame(df, frame):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting points for each marker
    for i, values in enumerate(df['Values']):
        x_value, y_value, z_value = values[frame]  # Extracting the value [x, y, z]
        ax.scatter(x_value, y_value, z_value, label=f'{df.Name[i]}', color='blue')  # Plotting the point for each marker

    # Define pairs of joints to connect
    joint_pairs = [('0_mid_Waist', '1_right_Waist'), ('1_right_Waist', '2_Right_knee'),
                   ('2_Right_knee', '3_Right_foot'), ('0_mid_Waist', '4_left_waist'),
                   ('4_left_waist', '5_left_knee'), ('5_left_knee', '6_left_foot'),
                   ('0_mid_Waist', '7_SpineThoracic'), ('7_SpineThoracic', '8_SpineThoracic'),
                   ('8_SpineThoracic', '9_Neck_Head'), ('8_SpineThoracic', '11_Left_Shoulder'),
                   ('11_Left_Shoulder', '12_left_elbow'), ('12_left_elbow', '13_left_wrist'),
                   ('8_SpineThoracic', '14_Right_Shoulder'), ('14_Right_Shoulder', '15_right_elbow'),
                   ('15_right_elbow', '16_right_wrist'),
                   ('9_Neck_Head', '10_Fronthead') ]

    # Plotting lines between corresponding points
    for pair in joint_pairs:
        marker1_values = df[df['Name'] == pair[0]]['Values'].iloc[0]
        marker2_values = df[df['Name'] == pair[1]]['Values'].iloc[0]
        x_values1, y_values1, z_values1 = marker1_values[frame]
        x_values2, y_values2, z_values2 = marker2_values[frame]
        ax.plot([x_values1, x_values2], [y_values1, y_values2], [z_values1, z_values2], color='red')

    # Set axis limits based on your data
    ax.set_xlim([-2000, 1000])
    ax.set_ylim([-200, 1000])
    ax.set_zlim([0, 2000])

    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=90, azim=0)

    # Adding labels and title
    ax.set_xlabel('X Value')
    ax.set_ylabel('Y Value')
    ax.set_zlabel('Z Value')
    plt.title('3D Plot for Each Marker')

    # Show legend
    ax.legend(loc='best')

    plt.show()
def plot_2d_points(df, frame):
    fig, ax = plt.subplots(figsize=(8, 8))
    num_points = len(df)
    colors = np.random.rand(num_points, 3)
    # Plotting points for each marker
    for i, values in enumerate(df['Projected Points']):
        x_value, z_value = values[frame]
        ax.scatter(x_value, z_value, color=colors[i], label=df.Name[i])  # Add label for legend

    # Define pairs of joints to connect
    joint_pairs = [('0_mid_Waist', '1_right_Waist'), ('1_right_Waist', '2_Right_knee'),
                   ('2_Right_knee', '3_Right_foot'), ('0_mid_Waist', '4_left_waist'),
                   ('4_left_waist', '5_left_knee'), ('5_left_knee', '6_left_foot'),
                   ('0_mid_Waist', '7_SpineThoracic'), ('7_SpineThoracic', '8_SpineThoracic'),
                   ('8_SpineThoracic', '9_Neck_Head'), ('8_SpineThoracic', '11_Left_Shoulder'),
                   ('11_Left_Shoulder', '12_left_elbow'), ('12_left_elbow', '13_left_wrist'),
                   ('8_SpineThoracic', '14_Right_Shoulder'), ('14_Right_Shoulder', '15_right_elbow'),
                   ('15_right_elbow', '16_right_wrist'),
                   ('9_Neck_Head', '10_Fronthead') ]

    # Plotting lines between corresponding points
    for pair in joint_pairs:
        marker1_values = df[df['Name'] == pair[0]]['Projected Points'].iloc[0]
        marker2_values = df[df['Name'] == pair[1]]['Projected Points'].iloc[0]
        x_values1, z_values1 = marker1_values[frame]
        x_values2, z_values2 = marker2_values[frame]
        ax.plot([x_values1, x_values2], [z_values1, z_values2], color='red')

    

    # Adding labels and title
    ax.set_xlabel('X Value')
    ax.set_ylabel('Z Value')
    ax.set_title('2D Plot for Each Marker')

    # Show legend
    ax.legend()

    plt.show()

DELTA_Y = 1036
DELTA_X = 839 
DEL_Y = 0.56
DEL_X = 0.52

def merge_points(*args):
    import numpy as np
    merged_points = []
    if len(args) < 2:
        raise ValueError("At least two sets of points are required")
    for points in zip(*args):
        merged_point = np.mean(points, axis=0)
        merged_points.append(merged_point.tolist())  
    return merged_points



def create_video_frames(file_head: str):
    frame_count = 0
    cap = cv2.VideoCapture("bigdata/videos/" + file_head + ".avi")
    out = 'result_images'
    os.makedirs(out, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(out, f'{file_head}_{frame_count:010d}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
        print(f'Frame {frame_count} saved')

    cap.release()


def export_pose_data(output_file, df_list):
    images = []
    annotations = []
    categories = [{'id': 1, 'name': 'person'}]  # Assuming category ID for a person is 1

    frame_name = 0
    

    for df in df_list:
        file_name = 19
        for frame_id in range(40,len(df['Values'][0]) - 40,2):
            frame_keypoints = []
            for index, row in df.iterrows():
                x_value,z_value = row['Values'][frame_id]

                #new_x = DELTA_X + DEL_X * x_value 
                #new_z = DELTA_Y - DEL_Y * z_value
                frame_keypoints.extend([x_value, z_value, 1 if 'left' in row['Name'].lower() else 2])  # Assuming confidence value is 2 for each keypoint
            
            bbox = calculate_bbox(frame_keypoints)

            annotations.append({
                #'segmentation': [],  # Add segmentation if available
                'keypoints': frame_keypoints,
                'num_keypoints': len(frame_keypoints) // 3,  # Assuming each keypoint has x, y, z, and confidence
                'area': bbox[2] * bbox[3],  # Add area if available
                'iscrowd': 0,  # Add iscrowd if available
                'image_id': frame_name,  # Assuming frame_id is unique for each image
                'bbox': bbox,  # Calculate bounding box
                'category_id': 1,  # Assuming category ID for a person is 1
                'id': frame_name  # Assuming frame_id is unique for each annotation
            })
            
            images.append({
                'file_name': f'{row["File"][0]}_{(file_name):010d}.jpg',  # Assuming image filenames are like '000000001268.jpg'
                'height': 1088,  # Add height if available
                'width': 1920,  # Add width if available
                'id': frame_name  # Assuming frame_id is unique for each image
            })
            frame_name += 1
            file_name += 1
    
    coco_data = {'images': images, 'annotations': annotations, 'categories': categories}
    
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)


def calculate_bbox(keypoints):
    x_coordinates = keypoints[::3]  # Extract x coordinates
    y_coordinates = keypoints[1::3]  # Extract y coordinates
    
    min_x = min(x_coordinates)
    min_y = min(y_coordinates)
    max_x = max(x_coordinates)
    max_y = max(y_coordinates)
    
    width = max_x - min_x
    height = max_y - min_y
    
    return [min_x, min_y, width, height]
    


def convert_to_mp4_120fps():
    # Step 1: Convert AVI to MP4 using ffmpeg command-line tool
    input_file =  'input.avi'
    output_file_mp4 = 'output_interpolated.mp4'
    
    # Run ffmpeg command to convert to MP4
    command = f"ffmpeg -i {input_file} {output_file_mp4}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("Conversion to MP4 failed")
        return

    # Check if conversion was successful
    if not os.path.exists(output_file_mp4):
        print("Conversion to MP4 failed")
        return

    # Step 2: Interpolate frame rate from 660 FPS to 120 FPS using ffmpeg command-line tool
    output_file_interpolated = 'output_interpolated.mp4'

    # Run ffmpeg command for frame rate interpolation
    command = f"ffmpeg -i {output_file_mp4} -vf minterpolate=fps=120 {output_file_interpolated}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        print("Frame rate interpolation failed")
        return

    print("Conversion and frame rate interpolation completed successfully")


def create_dataframe(file_head: str):
    with open("bigdata/json/" + file_head + ".json", 'r') as file:
        try:
            data = json.load(file)
        except UnicodeDecodeError:
            print("bigdata/json/" + file_head + ".json")
            raise


    markers_list = data.get('Markers', [])

    marker_names = []
    marker_values = []



    for marker in markers_list:
        marker_names.append(marker.get('Name', ''))
        parts = marker.get('Parts', [])
        values_list = []
        for part in parts:
            values = part.get('Values', [])

            values = [v[:-1] for v in values]
            values_list.extend(values)
        marker_values.append(values_list)

    camera_values = data.get('Cameras')[4]

    if not camera_values.get("Serial") == 28002:
        raise Exception('The chosen camera is not the right camera')
    
    camera_transform = camera_values.get('Transform')

    translation_vector = np.array([camera_transform.get(key) for key in ['x','y']])
    translation_vector = np.append(translation_vector, np.array([4.5*float(camera_transform.get('z'))])) # scaling factor

    camera_rotation = np.array([
        [camera_transform.get(key) for key in ['r11','r12','r13','r21','r22','r23','r31','r32','r33']]
    ]).reshape(3,3)

    camera_intrinsics = camera_values.get('Intrinsic')

    camera_focal_length = float(camera_intrinsics.get('FocalLength'))
    camera_center_point = [camera_intrinsics.get('CenterPoint' + i) for i in ['U', 'V']]

    camera_matrix = np.array([
        [camera_intrinsics.get('FocalLengthU'), 0, camera_intrinsics.get('CenterPointU')], 
        [0, camera_intrinsics.get('FocalLengthV'), camera_intrinsics.get('CenterPointV')],
        [0, 0, 64] 
    ]) / 64


    dist_coeffs = np.array(
        [camera_intrinsics.get(key) for key in ["RadialDistortion1","RadialDistortion2", "TangentalDistortion1","TangentalDistortion2"] ]
    )      
    #print(dist_coeffs)
    # Change for the training data
    origo = {}
    origo['runner'] = [1310, 1140]
    origo['runner'] = [1310, 1140]
    origo['runner'] = [1310, 1140]
    origo['runner'] = [1310, 1140]
    origo['runner'] = [1280, 1140]
    origo['runner'] = [1280, 1140]
    origo['runner'] = [1280, 1140]
    origo['runner'] = [1240, 1140]
    origo['runner'] = [1240, 1140]
    origo['runner'] = [1390, 1100]
    origo['runner'] = [1390, 1100]
    origo['runner'] = [1390, 1100]
    origo['runner'] = [1390, 1100]


    camera_settings = {
        'translation_vector' : translation_vector,
        'camera_rotation' : camera_rotation,
        'camera_matrix' : camera_matrix,
        'dist_coeffs' : dist_coeffs,
        'focal_length': camera_focal_length,
        'center_point': camera_center_point,
        'origin': origo
    }
    CAMERA_DISC[file_head] = camera_settings
    
    return pd.DataFrame({'File': file_head, 'Name': marker_names, 'Values': marker_values}) 



def create_merged_points_dataframe(df):
    df_merged_points = pd.DataFrame(columns=['File', 'Name', 'Values'])
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '0_Nose', 'Values': df['Values'][38]}, ignore_index=True)  # Might be a good idea to use one of the chest/top of back points weighted to take all the points on the face down a little. 
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '1_Left-eye', 'Values': merge_points(df['Values'][37] ,df['Values'][38])}, ignore_index=True)  # Weight?
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '2_Right-eye', 'Values': merge_points(df['Values'][36] ,df['Values'][38])}, ignore_index=True)  # Weight?
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '3_Left-ear', 'Values': df['Values'][37]}, ignore_index=True)  # Weight?
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '4_Right-ear', 'Values': df['Values'][36]}, ignore_index=True)  # Weight?
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '5_Left_shoulder', 'Values': df['Values'][34]}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '6_Right_shoulder', 'Values': df['Values'][33]}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '7_Left_elbow', 'Values': merge_points(df['Values'][26], df['Values'][28])}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '8_Right_elbow', 'Values': merge_points(df['Values'][25], df['Values'][27])}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '9_Left-wrist', 'Values': merge_points(df['Values'][19], df['Values'][17])}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '10_Right-wrist', 'Values': merge_points(df['Values'][18], df['Values'][16])}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '11_Left-hip', 'Values': merge_points(df['Values'][21], df['Values'][23])}, ignore_index=True)  # Weight?
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '12_Right-hip', 'Values': merge_points(df['Values'][20], df['Values'][22])}, ignore_index=True)  # Weight?
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '13_Left_knee', 'Values': merge_points(df['Values'][11],df['Values'][13])}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '14_Right-knee', 'Values': merge_points(df['Values'][10],df['Values'][12])}, ignore_index=True)
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '15_Left-ankle', 'Values': df['Values'][5]}, ignore_index=True)  # Might be better to add a weight and add in the heel.
    df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '16_Right-ankle', 'Values': df['Values'][4]}, ignore_index=True)  # same

    
    #df_merged_points = df_merged_points._append({'File':df['File'] ,'Name': '7_SpineThoracic', 'Values': df['Values'][31]}, ignore_index=True)
    
    
    

    return df_merged_points


def main():
    train_heads = [
        "runner_5", "runner_7.30", 
        "runner_6.40", 
        "runner_4.08", "runner_4.43",  
        "runner_5.13", "runner_7.30", 
        "runner_5.43", "runner_6", 
        "runner_3.38", "runner_4.08", "runner_5.27", 
        "runner_4.37", "runner_6", 
        "runner_4.17", "runner_5",  
        "runner_4", "runner_5", "runner_6", 
        "runner_4", "runner_5", 
        "runner_4.27", "runner_6.19", 
        "runner_4.37", "runner_5.43", 
        "runner_4.17", "runner_6.40"
    ]
    validation_heads = ["runner_3.45", "runner_5.43", "runner_4", "runner_6", "runner_6"]


    train_df_list = []
    validation_df_list = []

    for file_head in train_heads:
        # Import file
        print(file_head)
        df = create_dataframe(file_head)
        df_projected = project_2d(df)
        #print(df_projected)
        # Merge the 43 points to 17
        train_df_list.append(create_merged_points_dataframe(df_projected))
        #create_video_frames(file_head)
    
    for file_head in validation_heads:
        # Import file
        print(file_head)
        df = create_dataframe(file_head)
        df_projected = project_2d(df)
        #print(df_projected)
        # Merge the 43 points to 17
        validation_df_list.append(create_merged_points_dataframe(df_projected))
        #create_video_frames(file_head)
    
    #
    #projected_df = project_onto_xz_plane(df_merged_points)

    #plot_2d_points(projected_df, frame=16)

    # Plot the one frame
    #plot_single_frame(df_merged_points,16)

    #convert_to_mp4_120fps()

    export_pose_data('train4.json', train_df_list)
    export_pose_data('validation4.json', validation_df_list)


if __name__ == "__main__":
    main() 
















