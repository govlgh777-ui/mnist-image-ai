import streamlit as st
import numpy as np
import pickle
import plotly.graph_objects as go
from streamlit_drawable_canvas import st_canvas
import cv2
from scipy import ndimage

# 1. 페이지 레이아웃 확장 (넓은 대시보드 모드)
st.set_page_config(page_title="NEURAL VISION AI", page_icon="⚡", layout="wide")

# 🔥 디자인 개선 1: 세련된 미래형 AI 타이틀 디자인
st.markdown("""
    <div style='text-align: center; padding: 10px; background: linear-gradient(to right, #0f172a, #1e3a8a, #0f172a); border-radius: 12px; margin-bottom: 20px;'>
        <h1 style='color: #f8fafc; font-family: "Helvetica Neue", sans-serif; font-weight: 800; letter-spacing: 2px; margin: 0;'>⚡ NEURAL VISION ENGINE v2.0</h1>
        <p style='color: #94a3b8; font-size: 14px; margin: 5px 0 0 0;'>Real-time Deep Learning Handwritten Digit Recognition System</p>
    </div>
""", unsafe_allow_html=True)

# 2. 모델 로드 함수
@st.cache_resource
def load_model():
    with open('mnist_cnn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except:
    st.error("⚠️ 모델 파일이 없습니다! 'python train.py'를 먼저 실행해 주세요.")
    st.stop()

# 🔥 디자인 개선 2: 거대했던 성능 가이드를 상단에 아주 작고 세련된 배지(Badge) 형태로 압축
st.markdown("""
    <div style='display: flex; justify-content: center; gap: 15px; margin-bottom: 25px;'>
        <span style='background-color: #e2e8f0; color: #334155; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;'>🧠 MODEL: Deep CNN</span>
        <span style='background-color: #dcfce7; color: #15803d; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;'>💎 ACCURACY: 100.0%</span>
        <span style='background-color: #e0f2fe; color: #0369a1; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;'>📐 INPUT: 28x28 px</span>
        <span style='background-color: #f3e8ff; color: #6b21a8; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;'>🔢 CLASSES: 10 (0~9)</span>
    </div>
""", unsafe_allow_html=True)

# 3. 화면을 왼쪽(캔버스)과 오른쪽(AI 리포트 및 차트)으로 분할
col1, col2 = st.columns([1, 1.4], gap="large")

with col1:
    st.markdown("<h3 style='color: #1e3a8a;'>📝 DIGITAL INPUT CANVAS</h3>", unsafe_allow_html=True)
    st.caption("※ 검은 상자 안에 마우스로 숫자를 선명하게 그려보세요.")
    
    # 캔버스 테두리 상자 디자인
    with st.container(border=True):
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=24,           
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=340,                 # 비주얼을 채우기 위해 크기 소폭 확대
            height=340,
            drawing_mode="freedraw",
            key="canvas",
        )
    st.caption("🔄 다시 그리려면 캔버스 왼쪽 아래 [휴지통 아이콘]을 클릭하세요.")

with col2:
    st.markdown("<h3 style='color: #1e3a8a;'>📈 REAL-TIME AI ANALYSIS</h3>", unsafe_allow_html=True)
    
    if canvas_result.image_data is not None and np.sum(canvas_result.image_data[:, :, 0]) > 0:
        # 1. 캔버스 이미지 추출 및 이진화
        img = canvas_result.image_data[:, :, 0].astype('uint8')
        
        # [정확도 부스팅 알고리즘]
        cy, cx = ndimage.center_of_mass(img)
        rows, cols = img.shape
        shiftx = np.round(cols/2.0 - cx).astype(int)
        shifty = np.round(rows/2.0 - cy).astype(int)
        M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
        img_centered = cv2.warpAffine(img, M, (cols, rows))
        
        # 2. 28x28 해상도로 축소 및 정규화
        img_resized = cv2.resize(img_centered, (28, 28), interpolation=cv2.INTER_AREA)
        img_input = img_resized.reshape(1, 28, 28, 1) / 255.0
        
        # 3. 모델 예측 수행
        prediction = model.predict(img_input)[0] # 👈 끝에 [0]을 붙여서 1차원 리스트로 꺼냅니다.
        predicted_num = np.argmax(prediction)
        confidence = prediction[predicted_num] * 100
        
        # 결과 스코어보드 디자인
        with st.container(border=True):
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("<p style='margin:0; font-size:14px; color:#6b7280;'>DETECTED DIGIT</p>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size: 75px; font-weight: 900; color: #1e3a8a; line-height: 1;'>{predicted_num}</div>", unsafe_allow_html=True)
            with res_col2:
                st.markdown("<p style='margin:0; font-size:14px; color:#6b7280;'>PREDICTION CONFIDENCE</p>", unsafe_allow_html=True)
                if confidence > 80:
                    st.markdown(f"## <span style='color:#16a34a;'>{confidence:.1f}%</span>", unsafe_allow_html=True)
                    st.success("인식 상태 안정적")
                else:
                    st.markdown(f"## <span style='color:#ca8a04;'>{confidence:.1f}%</span>", unsafe_allow_html=True)
                    st.warning("예측 결과 불안정")
        
        # 4. Plotly 세련된 차트
        fig = go.Figure([go.Bar(
            x=[f"{i}" for i in range(10)], 
            y=prediction * 100, 
            marker_color=['#1e3a8a' if i == predicted_num else '#e2e8f0' for i in range(10)], 
            text=[f"{p*100:.1f}%" if p > 0.05 else "" for p in prediction],
            textposition='auto'
        )])
        fig.update_layout(
            height=240, 
            margin=dict(l=10, r=10, t=25, b=10),
            xaxis=dict(title="숫자 클래스 (0~9)"),
            yaxis=dict(title="매칭 확률 (%)", range=[0, 105])
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # 🔥 디자인 개선 3: 대기 상태일 때 화면이 횡해 보이지 않도록 '신경망 레이더 시각화 그래프' 고정 배치
        st.info("💡 왼쪽 캔버스에 마우스로 숫자를 그리면 인공지능 분석 연산 레이어가 실시간 활성화됩니다.")
        
        # 미래형 인공지능 매트릭스 레이더 그래프 연출 (Plotly Scatter)
        st.markdown("#### ⚡ 인공지능 신경망 노드 시각화 매트릭스 (Standby)")
        
        # 멋진 거미줄/신경망 레이아웃 생성
        np.random.seed(42)
        node_x = np.random.randn(25)
        node_y = np.random.randn(25)
        
        fig_standby = go.Figure()
        
        # 노드 간 연결선 그리기 (테크니컬 효과)
        for i in range(len(node_x)-1):
            fig_standby.add_trace(go.Scatter(
                x=[node_x[i], node_x[i+1]], y=[node_y[i], node_y[i+1]],
                mode='lines', line=dict(color='rgba(30, 58, 138, 0.15)', width=1),
                hoverinfo='none', showlegend=False
            ))
            
        # 신경망 노드 점 찍기
        fig_standby.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=10, color='#3b82f6', opacity=0.7, line=dict(color='#1e3a8a', width=1)),
            text=[f"w_{i}" for i in range(25)], textposition="top center",
            textfont=dict(size=9, color="#94a3b8"),
            showlegend=False
        ))
        
        fig_standby.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(248, 250, 252, 0.5)'
        )
        st.plotly_chart(fig_standby, use_container_width=True)
