import gradio as gr
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from tensorflow.keras.models import load_model

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile

# ================= LOAD MODEL =================
model = load_model("final_pregnancy_dl_model.keras")
scaler = joblib.load("final_scaler.pkl")
le = joblib.load("final_label_encoder.pkl")

# ================= HELPERS =================

def classify_tsh(tsh):
    if tsh < 0.1: return "Hyperthyroid"
    elif tsh <= 2.5: return "Optimal"
    elif tsh <= 4.0: return "Subclinical Hypothyroid"
    return "Overt Hypothyroid"

def classify_glucose(g):
    if g < 85: return "Optimal"
    elif g <= 91: return "High Normal"
    elif g <= 110: return "GDM Risk"
    return "Diabetes Range"

def classify_hemoglobin(hb):
    if hb >= 11: return "Normal"
    elif hb >= 10: return "Mild Anemia"
    elif hb >= 7: return "Moderate Anemia"
    return "Severe Anemia"

# ================= FOOD =================

def food_recommendation(risk, hb, sugar, tsh):
    tips = []

    if hb < 7:
        tips.append("<b>Severe anemia</b>: urgent medical care, iron supplements, dates, spinach, beetroot.")
    elif hb < 10:
        tips.append("<b>Moderate anemia</b>: include iron-rich foods like leafy greens, jaggery, lentils.")
    elif hb < 11:
        tips.append("<b>Mild anemia</b>: increase iron intake with nuts, seeds, and green vegetables.")

    if sugar > 110:
        tips.append("<b>High blood sugar</b>: avoid sweets, white rice, fried food. Prefer whole grains and fiber.")
    elif sugar >= 92:
        tips.append("<b>Borderline glucose</b>: limit sugar intake and monitor diet carefully.")

    if tsh > 4:
        tips.append("<b>High TSH</b>: include iodine-rich foods like dairy, eggs, and iodized salt.")
    elif tsh < 0.1:
        tips.append("<b>Low TSH</b>: avoid excess iodine and consult doctor.")

    if risk == "High Risk":
        tips.append("<b>High-risk pregnancy</b>: regular doctor visits, balanced rest, and strict diet control.")
    elif risk == "Medium Risk":
        tips.append("<b>Moderate risk</b>: maintain healthy nutrition and periodic monitoring.")
    else:
        tips.append("<b>Low risk</b>: continue balanced pregnancy diet.")

    return "### Personalized Nutrition Advice\n\n" + "\n\n".join(f"• {t}" for t in tips)

# ================= PDF REPORT =================
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import tempfile
from datetime import datetime

def generate_report(risk, food, clinical):

    from datetime import datetime
    file_path = datetime.now().strftime("Synopsis_%Y%m%d_%H%M%S.pdf")
    doc = SimpleDocTemplate(file_path)

    styles = getSampleStyleSheet()
    story = []

    # ================= STYLES =================
    title_style = ParagraphStyle(
        'title',
        parent=styles['Title'],
        fontSize=20,
        alignment=1,
        textColor=colors.darkblue
    )

    subtitle_style = ParagraphStyle(
        'subtitle',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1,
        textColor=colors.grey
    )

    section_style = ParagraphStyle(
        'section',
        parent=styles['Heading2'],
        textColor=colors.darkgreen
    )

    normal_style = ParagraphStyle(
        'normal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=6
    )

    # ================= HEADER =================
    story.append(Paragraph("<b>Pregnancy Risk Analysis</b>", title_style))
    story.append(Paragraph("Report", subtitle_style))

    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Spacer(1, 10))

    # Date
    today = datetime.now().strftime("%d %B %Y")
    story.append(Paragraph(f"<b>Date:</b> {today}", normal_style))
    story.append(Spacer(1, 10))

    # ================= MAIN BOX =================

    if "High" in risk:
        risk_color = "red"
    elif "Medium" in risk:
        risk_color = "orange"
    else:
        risk_color = "green"

    clean_food = food.replace("###", "").replace("\n", "<br/>")
    clean_clinical = clinical.replace("<br>", "<br/>")

    main_table = Table([
        [Paragraph(f"<b>Pregnancy Risk:</b> <font color='{risk_color}'>{risk}</font>", normal_style)],
        [Paragraph("<b>Nutrition Suggestion</b>", section_style)],
        [Paragraph(clean_food, normal_style)],
        [Paragraph("<b>Status</b>", section_style)],
        [Paragraph(clean_clinical, normal_style)]
    ], colWidths=[6*inch])

    main_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 2, colors.black),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,1), (-1,1), colors.lightgrey),
        ('BACKGROUND', (0,3), (-1,3), colors.lightgrey),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))

    story.append(main_table)
    story.append(Spacer(1, 20))

    # ================= HEALTH SUMMARY BOX =================

    summary_table = Table([
        [Paragraph("<b>Health Summary</b>", section_style)],

        [Paragraph(
            "Based on the provided inputs, key health indicators such as hemoglobin, glucose, and thyroid levels were analyzed to determine pregnancy risk.",
            normal_style
        )],

        [Paragraph("<b>Key Recommendations</b>", section_style)],

        [Paragraph(
            "• Maintain a balanced and nutritious diet.<br/>"
            "• Attend regular medical check-ups.<br/>"
            "• Monitor health parameters consistently.<br/>"
            "• Ensure proper rest and hydration.",
            normal_style
        )],

        [Paragraph("<b>Important Advice</b>", section_style)],

        [Paragraph(
            "If symptoms like bleeding, severe pain, or dizziness occur, seek immediate medical attention.",
            normal_style
        )]
    ], colWidths=[6*inch])

    summary_table.setStyle(TableStyle([
        ('BOX', (0,0), (-1,-1), 2, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightblue),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 20))

    # ================= DISCLAIMER =================

    story.append(Paragraph(
        "<font size=9 color=grey><b>Note:</b> This report is intended to assist in understanding potential pregnancy health risks. "
        "It is not a confirmed medical diagnosis. Please consult a qualified healthcare professional for further evaluation and treatment.</font>",
        styles['Normal']
    ))

    doc.build(story)

    return file_path
# ================= CHARTS =================
def create_gauge(value, title, min_val, max_val):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': f"<b>{title}</b>"},
        gauge={
            'axis': {'range': [min_val, max_val]},
            
            # needle color
            'bar': {'color': "blue"},

            # colored zones
            'steps': [
                {'range': [min_val, (min_val+max_val)*0.5], 'color': "lightgreen"},
                {'range': [(min_val+max_val)*0.5, (min_val+max_val)*0.75], 'color': "orange"},
                {'range': [(min_val+max_val)*0.75, max_val], 'color': "red"},
            ],
        }
    ))

    fig.update_layout(height=250)

    return fig
def create_probability_chart(probs):
    colors = ["green", "orange", "red"]  # Low, Medium, High

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=le.classes_,
        y=probs,
        marker_color=colors
    ))

    fig.update_layout(
        title="Risk Probability Distribution",
        yaxis_title="Probability",
        xaxis_title="Risk Class",
        height=300
    )

    return fig
# ================= PREDICTION =================

def predict_all(*inputs):

    cols = [
        "Age","BMI","Previous_Miscarriage","Previous_Pregnancies",
        "Systolic_BP","Diastolic_BP","Heart_Rate","Body_Temp",
        "CRL_mm","HB","Sugar","TSH","HCG","Progesterone",
        "Gestational_Week","Sac_Size_mm","CRL_US_mm",
        "Heartbeat_Presence_Yes","Bleeding_Yes","Abdominal_Pain_Yes",
        "Vomiting_Yes","Dizziness_Yes"
    ]

    clean_inputs = [0 if v is None else v for v in inputs]
    df = pd.DataFrame([dict(zip(cols, clean_inputs))])

    scaled = scaler.transform(df)
    pred = model.predict(scaled)
    probs = pred[0]

    risk = le.inverse_transform([np.argmax(probs)])[0]

    clinical = (
        f"<b>Thyroid:</b> {classify_tsh(df['TSH'][0])}<br><br>"
        f"<b>Glucose:</b> {classify_glucose(df['Sugar'][0])}<br><br>"
        f"<b>Hemoglobin:</b> {classify_hemoglobin(df['HB'][0])}"
    )

    return (
        risk,
        food_recommendation(risk, df['HB'][0], df['Sugar'][0], df['TSH'][0]),
        clinical,
        create_gauge(df['TSH'][0], "TSH", 0, 10),
        create_gauge(df['Sugar'][0], "Glucose", 60, 200),
        create_gauge(df['HB'][0], "Hemoglobin", 5, 15),
        create_probability_chart(probs)
    )

# ================= TAB VISIBILITY =================

def show_prediction():
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def show_nutrition():
    return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def show_clinical():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)

def show_charts():
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)

# ================= UI =================

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
        footer {visibility: hidden;}
        .gradio-container .footer {display: none !important;}
    """
) as demo:

    gr.Markdown("# Pregnancy Risk & Nutrition Dashboard")

    with gr.Column() as input_page:
        inputs = [gr.Number(label=l) for l in [
            "Age","BMI","Prev Miscarriage","Prev Pregnancies",
            "Sys BP","Dia BP","Heart Rate","Body Temp",
            "CRL","HB","Glucose","TSH","HCG","Progesterone",
            "Week","Sac Size","CRL US",
            "Heartbeat","Bleeding","Pain","Vomiting","Dizziness"
        ]]
        start_btn = gr.Button("Start Prediction")

    with gr.Column(visible=False) as main_page:

        with gr.Row():
            btn_pred = gr.Button("Prediction")
            btn_food = gr.Button("Nutrition ")
            btn_clin = gr.Button("Status")
            btn_chart = gr.Button("Charts")
            btn_download = gr.Button("Download")  # ✅ your button

        with gr.Column(visible=True) as sec_pred:
            risk_out = gr.Textbox(label="Pregnancy Risk")

        with gr.Column(visible=False) as sec_food:
            food_out = gr.Markdown(label="Nutrition Recommendation")

        with gr.Column(visible=False) as sec_clin:
            clinical_out = gr.Markdown(label="Clinical Status")

        with gr.Column(visible=False) as sec_chart:
            tsh_plot = gr.Plot()
            glu_plot = gr.Plot()
            hb_plot = gr.Plot()
            prob_plot = gr.Plot()

        # ✅ DOWNLOAD COMPONENT (Hugging Face correct way)
        report_file = gr.File(label="Download", visible=False)

    start_btn.click(
        predict_all,
        inputs=inputs,
        outputs=[risk_out, food_out, clinical_out, tsh_plot, glu_plot, hb_plot, prob_plot]
    ).then(lambda: [gr.update(visible=False), gr.update(visible=True)], None, [input_page, main_page])

    btn_pred.click(show_prediction, outputs=[sec_pred, sec_food, sec_clin, sec_chart, report_file])
    btn_food.click(show_nutrition, outputs=[sec_pred, sec_food, sec_clin, sec_chart, report_file])
    btn_clin.click(show_clinical, outputs=[sec_pred, sec_food, sec_clin, sec_chart, report_file])
    btn_chart.click(show_charts, outputs=[sec_pred, sec_food, sec_clin, sec_chart, report_file])

    # ✅ FINAL DOWNLOAD (WORKING)
    btn_download.click(
        generate_report,
        inputs=[risk_out, food_out, clinical_out],
        outputs=report_file
    ).then(lambda: gr.update(visible=True), None, report_file)

demo.launch(server_name="0.0.0.0",server_port=10000)
