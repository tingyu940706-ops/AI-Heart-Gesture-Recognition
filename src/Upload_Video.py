import os
import streamlit as st
import cv2
import tempfile
from heart_pose import process_heart_pose_frame

st.title('Heart Gesture Detector (Upload Video)')

output_video_file = 'output_recorded.mp4'

if os.path.exists(output_video_file):
    os.remove(output_video_file)

with st.form('Upload', clear_on_submit=True):
    up_file = st.file_uploader("Upload a Video", ['mp4','mov', 'avi'])
    uploaded = st.form_submit_button("Upload")

stframe = st.empty()
warn = st.empty()
download_button = st.empty()

if up_file and uploaded:
    download_button.empty()
    tfile = tempfile.NamedTemporaryFile(delete=False)

    try:
        warn.empty()
        tfile.write(up_file.read())
        vf = cv2.VideoCapture(tfile.name)

        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_output = cv2.VideoWriter(output_video_file, fourcc, fps, frame_size)

        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out_frame, _ = process_heart_pose_frame(frame)
            stframe.image(out_frame)
            video_output.write(out_frame[..., ::-1])

        vf.release()
        video_output.release()
        stframe.empty()
        tfile.close()

    except AttributeError:
        warn.markdown("Please upload a valid video file.")

if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data=op_vid, file_name='output_recorded.mp4')

    if download:
        st.session_state['download'] = True

if os.path.exists(output_video_file) and st.session_state.get('download', False):
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()
