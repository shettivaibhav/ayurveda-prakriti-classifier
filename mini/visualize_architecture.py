"""
Interactive Architecture Diagram for Wrist Image Dosha Classification
Uses Plotly to create a visual flow diagram of the entire system
"""

import plotly.graph_objects as go

# Define node positions and properties
nodes = {
    # Data Layer
    'data': {'pos': (0.5, 9), 'color': '#4ECDC4', 'label': 'DATA LAYER\n~80 subjects × 3 classes\n(Vata/Pitta/Kapha)\npalm_open_back.jpg'},
    
    # Preprocessing
    'preprocess': {'pos': (0.5, 7.5), 'color': '#45B7D1', 'label': 'PREPROCESSING\nResize: 224×224\nAugmentation: Rotation, Flip\nNormalize: ImageNet stats'},
    
    # Dataset
    'dataset': {'pos': (0.5, 6), 'color': '#95E1D3', 'label': 'WristDataset\nCustom PyTorch Dataset\n80/20 train/val split'},
    
    # DataLoader
    'dataloader': {'pos': (0.5, 4.5), 'color': '#F38181', 'label': 'DataLoader\nBatch size: 16\nShuffle + Pin memory'},
    
    # Model
    'mobilenet': {'pos': (0.5, 3), 'color': '#AA96DA', 'label': 'MobileNetV2 Backbone\n3.5M parameters\nPre-trained ImageNet'},
    
    'classifier': {'pos': (0.5, 1.5), 'color': '#FCBAD3', 'label': 'Custom Head\nDropout(0.2) → Linear(1280→256)\nReLU → Dropout(0.3)\nLinear(256→3)'},
    
    # Training components (left side)
    'loss': {'pos': (0.15, 0), 'color': '#FFD93D', 'label': 'Loss\nCrossEntropy'},
    
    'optimizer': {'pos': (0.35, -1.2), 'color': '#FFEAA7', 'label': 'Optimizer\nAdam lr=1e-4'},
    
    'scheduler': {'pos': (0.15, -2.4), 'color': '#FD79A8', 'label': 'Scheduler\nReduceLROnPlateau'},
    
    # Validation (right side)
    'validation': {'pos': (0.85, 0), 'color': '#74B9FF', 'label': 'Validation\nAccuracy + F1'},
    
    'early_stop': {'pos': (0.85, -1.2), 'color': '#A29BFE', 'label': 'Early Stopping\nPatience: 10'},
    
    # Model checkpoint
    'checkpoint': {'pos': (0.5, -3.5), 'color': '#6C5CE7', 'label': 'MODEL CHECKPOINT\nmodel.pth\nWeights + Metadata'},
    
    # Inference (bottom)
    'inference_cli': {'pos': (0.25, -5), 'color': '#00B894', 'label': 'CLI Inference\ninference.py\nCommand-line tool'},
    
    'inference_web': {'pos': (0.75, -5), 'color': '#00CEC9', 'label': 'Web App\napp.py (Streamlit)\nUpload & Classify'},
    
    # Classification logic
    'classification': {'pos': (0.5, -6.5), 'color': '#FF7675', 'label': 'Dosha Classification\nSingle/Dual/Tri-Dosha\nThreshold: 15%'},
    
    # Output
    'output': {'pos': (0.5, -8), 'color': '#55EFC4', 'label': 'OUTPUT\nDosha Type + Confidence\nProbability Distribution'}
}

# Define edges (connections)
edges = [
    ('data', 'preprocess'),
    ('preprocess', 'dataset'),
    ('dataset', 'dataloader'),
    ('dataloader', 'mobilenet'),
    ('mobilenet', 'classifier'),
    ('classifier', 'loss'),
    ('loss', 'optimizer'),
    ('optimizer', 'scheduler'),
    ('classifier', 'validation'),
    ('validation', 'early_stop'),
    ('scheduler', 'checkpoint'),
    ('early_stop', 'checkpoint'),
    ('checkpoint', 'inference_cli'),
    ('checkpoint', 'inference_web'),
    ('inference_cli', 'classification'),
    ('inference_web', 'classification'),
    ('classification', 'output'),
]

# Create figure
fig = go.Figure()

# Add edges
for edge in edges:
    x0, y0 = nodes[edge[0]]['pos']
    x1, y1 = nodes[edge[1]]['pos']
    
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode='lines',
        line=dict(color='#2D3436', width=2),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add arrow
    fig.add_annotation(
        x=x1, y=y1,
        ax=x0, ay=y0,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#2D3436',
    )

# Add nodes
for node_name, props in nodes.items():
    x, y = props['pos']
    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode='markers+text',
        marker=dict(
            size=80,
            color=props['color'],
            line=dict(color='white', width=2)
        ),
        text=props['label'],
        textposition='middle center',
        textfont=dict(size=9, color='black', family='Arial Black'),
        hoverinfo='text',
        hovertext=props['label'],
        showlegend=False
    ))

# Add section labels
sections = [
    {'text': '📊 DATA PIPELINE', 'x': 0.1, 'y': 9.5, 'color': '#2D3436'},
    {'text': '🧠 MODEL', 'x': 0.1, 'y': 3.5, 'color': '#2D3436'},
    {'text': '⚙️ TRAINING', 'x': 0.05, 'y': -1, 'color': '#2D3436'},
    {'text': '✅ VALIDATION', 'x': 0.88, 'y': -1, 'color': '#2D3436'},
    {'text': '🚀 DEPLOYMENT', 'x': 0.1, 'y': -5.5, 'color': '#2D3436'},
]

for section in sections:
    fig.add_annotation(
        text=section['text'],
        x=section['x'],
        y=section['y'],
        showarrow=False,
        font=dict(size=14, color=section['color'], family='Arial Black'),
        xanchor='left'
    )

# Update layout
fig.update_layout(
    title={
        'text': '🖐️ Wrist Image Dosha Classification - System Architecture',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': '#2D3436', 'family': 'Arial Black'}
    },
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-0.1, 1.1]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[-8.5, 10]
    ),
    plot_bgcolor='#F8F9FA',
    width=1200,
    height=1400,
    margin=dict(l=50, r=50, t=100, b=50)
)

# Add legend box
fig.add_annotation(
    text=(
        '<b>Legend:</b><br>'
        '🔵 Data Processing | 🟣 Model Components | 🟡 Training | 🟢 Inference<br>'
        '<br>'
        '<b>Key Metrics:</b><br>'
        '• Dataset: ~80 samples (3 classes)<br>'
        '• Model: MobileNetV2 (3.5M params)<br>'
        '• Training: Adam, lr=1e-4, batch=16<br>'
        '• Deployment: CLI + Streamlit Web App'
    ),
    x=0.5,
    y=-8.8,
    showarrow=False,
    font=dict(size=10, color='#2D3436'),
    bgcolor='white',
    bordercolor='#2D3436',
    borderwidth=2,
    xanchor='center'
)

# Save as HTML
fig.write_html('architecture_diagram.html')
print("✅ Interactive architecture diagram saved as 'architecture_diagram.html'")
print("📊 Open it in your browser to explore the system architecture!")

# Show in default browser
fig.show()
