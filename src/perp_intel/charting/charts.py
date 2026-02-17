import altair as alt

def altair_line_chart(df, x, x_title, y, y_title, tooltip):

    chart = alt.Chart(df).mark_line().encode(
        x=alt.X(x, title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y(y, title=y_title, scale=alt.Scale(zero=False)),
        tooltip=tooltip
    ).properties(height=350).interactive()

    return chart