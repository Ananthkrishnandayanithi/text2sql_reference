import plotly.express as px
import pandas as pd

import plotly.graph_objects as go
import pandas as pd

data = [('Quality Control', 89), ('Major Mfg Projects', 8), ('Manufacturing', 140), ('Product Development', 34), ('Sales', 20), ('Account Management', 84), ('Green Building', 8), ('IT', 40), ('Facilities/Engineering', 58), ('Marketing', 48), ('Manufacturing Admin', 5), ('Training', 16), ('Quality Assurance', 67), ('Professional Training Group', 14), ('Environmental Compliance', 13), ('Creative', 19), ('Research/Development', 5), ('Environmental Health/Safety', 9), ('Human Resources', 7), ('Research Center', 5)]
df = pd.DataFrame(data, columns=['Department', 'Employee Count'])
df = df.sort_values('Employee Count')

fig = go.Figure(data=[go.Bar(x=df['Department'], y=df['Employee Count'])])
fig.update_layout(title_text='Employee Count by Department',xaxis_title="Department",yaxis_title="Employee Count")
fig.write_image("employee_count_bar_chart.png")
fig.write_image("static/plot.png")
