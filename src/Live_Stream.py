import av
import os
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes
from aiortc.contrib.media import MediaRecorder
from heart_pose import process_heart_pose_frame

st.title('Heart Gesture Detector (Live)')

output_video_file = 'output_live.flv'

def video_frame_callback(frame: av.VideoFrame):
    frame = frame.to_ndarray(format="bgr24")
    frame, _ = process_heart_pose_frame(frame)
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

def out_recorder_factory():
    return MediaRecorder(output_video_file)

ctx = webrtc_streamer(
    key="heart-gesture",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": {"width": {"min": 480, "ideal": 480}}, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
    out_recorder_factory=out_recorder_factory
)

download_button = st.empty()

if os.path.exists(output_video_file):
    with open(output_video_file, 'rb') as op_vid:
        download = download_button.download_button('Download Video', data=op_vid, file_name='output_live.flv')

        if download:
            st.session_state['download'] = True

if os.path.exists(output_video_file) and st.session_state.get('download', False):
    os.remove(output_video_file)
    st.session_state['download'] = False
    download_button.empty()
