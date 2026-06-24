import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
import os
import time
import tempfile
from scipy import interpolate 
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & HẰNG SỐ
# ==========================================
st.set_page_config(page_title="Nhận Diện Ngôn Ngữ Ký Hiệu", layout="wide")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28), 
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]

SEQUENCE_LENGTH = 60
CONFIDENCE_THRESHOLD = 0.50 

# ==========================================
# 2. CACHE MODELS & LABELS
# ==========================================
@st.cache_data
def load_labels_from_folders(dataset_dir):
    if not os.path.exists(dataset_dir):
        return []
    subfolders = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    return sorted(subfolders)

@st.cache_resource
def load_bilstm_model(model_path):
    try:
        return load_model(model_path, compile=False)
    except Exception as e:
        return None

# ==========================================
# 3. CÁC HÀM HỖ TRỢ XỬ LÝ ẢNH & MODEL
# ==========================================
def draw_vietnamese_text(img_rgb, text, position, text_color=(255, 255, 255), font_size=30):
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)
    try: 
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError: 
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=text_color)
    return np.array(img_pil)

def build_frame_vector(pose_res, hand_res):
    pose = np.zeros(75, dtype=np.float32)
    if pose_res.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in pose_res.pose_landmarks[0][:25]], dtype=np.float32).flatten()
        
    lh, rh = np.zeros(63, dtype=np.float32), np.zeros(63, dtype=np.float32)
    if hand_res.hand_landmarks:
        for i, hand in enumerate(hand_res.handedness):
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_res.hand_landmarks[i]], dtype=np.float32).flatten()
            if hand[0].category_name == 'Left': lh = pts
            else: rh = pts
            
    return np.concatenate([pose, lh, rh])

def draw_landmarks_on_image(frame, pose_res, hand_res):
    h, w, _ = frame.shape
    if pose_res.pose_landmarks:
        for pose_landmarks in pose_res.pose_landmarks:
            for connection in POSE_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                    start_lm, end_lm = pose_landmarks[start_idx], pose_landmarks[end_idx]
                    cv2.line(frame, (int(start_lm.x * w), int(start_lm.y * h)), (int(end_lm.x * w), int(end_lm.y * h)), (0, 255, 0), 2)
            for lm in pose_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)

    if hand_res.hand_landmarks:
        for hand_landmarks in hand_res.hand_landmarks:
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start_lm, end_lm = hand_landmarks[start_idx], hand_landmarks[end_idx]
                    cv2.line(frame, (int(start_lm.x * w), int(start_lm.y * h)), (int(end_lm.x * w), int(end_lm.y * h)), (255, 165, 0), 2)
            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (255, 255, 255), -1)
    return frame

def sample_sequence_by_interpolation(sequence, target_length=60):
    seq_array = np.array(sequence)
    T = len(seq_array)
    
    if T == 0: return np.zeros((target_length, 201), dtype=np.float32)
    if T == target_length: return seq_array.astype(np.float32)
    if T == 1: return np.tile(seq_array, (target_length, 1)).astype(np.float32)
        
    f = interpolate.interp1d(np.linspace(0, 1, T), seq_array, axis=0, kind='linear')
    return f(np.linspace(0, 1, target_length)).astype(np.float32)

# ==========================================
# 4. CẤU HÌNH SIDEBAR VÀ KHỞI TẠO
# ==========================================
st.title("Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu")

st.sidebar.header("Cấu Hình Đường Dẫn")
MODEL_PATH = st.sidebar.text_input("Keras Model Path", r'D:\sign_language_project\vsl_bilstm_model.h5')
TRAIN_DIR = st.sidebar.text_input("Dataset Folder (Labels)", r'D:\sign_language_project\hand-sign-recognition\data\Processed')
POSE_MODEL_PATH = st.sidebar.text_input("Pose Task Path", r'D:\sign_language_project\hand-sign-recognition\notebooks\models\pose_landmarker.task')
HAND_MODEL_PATH = st.sidebar.text_input("Hand Task Path", r'D:\sign_language_project\hand-sign-recognition\notebooks\models\hand_landmarker.task')

LABELS = load_labels_from_folders(TRAIN_DIR)
bilstm_model = load_bilstm_model(MODEL_PATH)

if not LABELS or bilstm_model is None:
    st.warning("⚠️ Vui lòng kiểm tra lại đường dẫn ở Sidebar để tải mô hình và nhãn.")
    st.stop()

st.sidebar.success(f"✅ Đã tải {len(LABELS)} nhãn và Model BiLSTM.")

# CHIA TABS GIAO DIỆN
tab1, tab2 = st.tabs(["🎥 Nhận Diện Qua Webcam", "📁 Nhận Diện Qua File Video"])

# ==========================================
# TAB 1: WEBCAM REAL-TIME
# ==========================================
with tab1:
    st.markdown("""
    **Hướng dẫn:** Hệ thống sẽ thu thập chuyển động của bạn liên tục. Mỗi **4 giây**, mô hình sẽ phân tích chuỗi hành động và đưa ra kết quả. Hãy thực hiện ký hiệu trong khung hình.
    """)
    
    col_btn1, col_btn2 = st.columns([1, 4])
    run_webcam = col_btn1.checkbox("🟢 Bật Webcam")
    clear_btn = col_btn2.button("🧹 Xóa câu")
    
    if 'cam_sentence' not in st.session_state or clear_btn:
        st.session_state['cam_sentence'] = []

    frame_window = st.image([])

    if run_webcam:
        # Sử dụng IMAGE mode cho Webcam để không bị lỗi timestamp khi livestream
        BaseOptions = python.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        opts_pose_img = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH), 
            running_mode=VisionRunningMode.IMAGE, num_poses=1)
            
        opts_hand_img = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH), 
            running_mode=VisionRunningMode.IMAGE, num_hands=2)

        cap = cv2.VideoCapture(0)
        sequence = []
        start_time = time.time()
        current_prediction = "Đang chờ dữ liệu..."
        
        with vision.PoseLandmarker.create_from_options(opts_pose_img) as pose_lmk, \
             vision.HandLandmarker.create_from_options(opts_hand_img) as hand_lmk:
             
            while run_webcam and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # Mirror frame
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                pose_res = pose_lmk.detect(mp_img)
                hand_res = hand_lmk.detect(mp_img)
                
                sequence.append(build_frame_vector(pose_res, hand_res))
                
                # Tính thời gian để gom batch (mỗi 4 giây dự đoán 1 lần)
                elapsed_time = time.time() - start_time
                remaining_time = max(0, 4.0 - elapsed_time)
                
                if elapsed_time >= 4.0:
                    if len(sequence) >= 10: # Cần đủ khung hình tối thiểu
                        sampled_seq = sample_sequence_by_interpolation(sequence, SEQUENCE_LENGTH)
                        input_data = np.expand_dims(sampled_seq, axis=0)
                        
                        res = bilstm_model.predict(input_data, verbose=0)[0]
                        best_idx = np.argmax(res)
                        conf = res[best_idx]
                        
                        if conf > CONFIDENCE_THRESHOLD:
                            word = LABELS[best_idx]
                            current_prediction = f"{word} ({conf*100:.1f}%)"
                            # Thêm vào câu hiển thị (giữ 5 từ gần nhất)
                            st.session_state['cam_sentence'].append(word)
                            if len(st.session_state['cam_sentence']) > 5:
                                st.session_state['cam_sentence'] = st.session_state['cam_sentence'][-5:]
                        else:
                            current_prediction = "Chưa rõ..."
                    else:
                        current_prediction = "Thiếu khung hình"
                        
                    # Reset lại bộ thu
                    sequence = []
                    start_time = time.time()

                # Vẽ UI lên khung hình
                rgb_frame = draw_landmarks_on_image(rgb_frame, pose_res, hand_res)
                
                h, w, _ = rgb_frame.shape
                cv2.rectangle(rgb_frame, (0, h - 80), (w, h), (0, 0, 0), -1)
                
                rgb_frame = draw_vietnamese_text(rgb_frame, f"Đếm ngược: {remaining_time:.1f}s", (10, 10), font_size=24, text_color=(255, 100, 100))
                rgb_frame = draw_vietnamese_text(rgb_frame, f"Dự đoán cuối: {current_prediction}", (10, 40), font_size=24, text_color=(100, 255, 100))
                
                sentence_str = " ".join(st.session_state['cam_sentence'])
                rgb_frame = draw_vietnamese_text(rgb_frame, f"Câu: {sentence_str}", (10, h - 60), font_size=32, text_color=(255, 255, 255))
                
                # Cập nhật Streamlit
                frame_window.image(rgb_frame, width=640)
                
        cap.release()

# ==========================================
# TAB 2: XỬ LÝ FILE VIDEO 
# ==========================================
with tab2:
    st.markdown("**Tải lên file video (.mp4, .avi, .mov) đã quay sẵn để phân tích.**")
    uploaded_video = st.file_uploader("Chọn File Video", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        col1, col2 = st.columns([1, 1])
        with col1:
            st.video(video_path)

        if st.button("Phân tích Video"):
            st.markdown("---")
            progress_bar = st.progress(0)
            status_text = st.empty()
            video_window = st.empty()
            
            # Sử dụng VIDEO mode cho File
            BaseOptions = python.BaseOptions
            opts_pose_vid = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH), 
                running_mode=vision.RunningMode.VIDEO, num_poses=1)
            opts_hand_vid = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH), 
                running_mode=vision.RunningMode.VIDEO, num_hands=2)

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 
            
            full_sequence = []
            processed_frames = [] 
            frame_idx = 0 
            
            status_text.text("Trạng thái: Đang trích xuất đặc trưng...")

            with vision.PoseLandmarker.create_from_options(opts_pose_vid) as pose_lmk, \
                 vision.HandLandmarker.create_from_options(opts_hand_vid) as hand_lmk:
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    ts_ms = int(frame_idx * 1000 / fps)
                    
                    pose_res = pose_lmk.detect_for_video(mp_img, ts_ms)
                    hand_res = hand_lmk.detect_for_video(mp_img, ts_ms)
                    
                    full_sequence.append(build_frame_vector(pose_res, hand_res))
                    
                    rgb_frame = draw_landmarks_on_image(rgb_frame, pose_res, hand_res)
                    processed_frames.append(rgb_frame)
                    
                    video_window.image(rgb_frame, use_container_width=True)
                    
                    frame_idx += 1 
                    if total_frames > 0:
                        progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    
            cap.release()

            status_text.text("Trạng thái: Đang dự đoán...")
            final_prediction_text = "Không nhận diện được"
            
            if len(full_sequence) > 0:
                sampled_sequence = sample_sequence_by_interpolation(full_sequence, SEQUENCE_LENGTH)
                input_data = np.expand_dims(sampled_sequence, axis=0)
                
                res = bilstm_model.predict(input_data, verbose=0)[0]
                best_class_idx = np.argmax(res)
                confidence = res[best_class_idx]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    final_prediction_text = f"{LABELS[best_class_idx]} ({confidence*100:.1f}%)"
            
            st.success(f"🎯 Kết quả: **{final_prediction_text}**")
            status_text.text("Hoàn thành!")
            
            st.markdown("### Xem lại video kết quả:")
            replay_window = st.empty()
            
            for frame_rgb in processed_frames:
                h, w, _ = frame_rgb.shape
                cv2.rectangle(frame_rgb, (0, 0), (w, 60), (0, 0, 0), -1)
                frame_rgb = draw_vietnamese_text(frame_rgb, final_prediction_text, position=(20, 15), font_size=35)
                
                replay_window.image(frame_rgb, use_container_width=True)
                time.sleep(1/fps)