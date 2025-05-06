import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Added for heatmap visualization

def data_visualization_page():
    st.title("Data Visualization")
    st.write("Select a column to visualize its distribution.")

    if 'df' in st.session_state:
        df = st.session_state.df

        st.write("Data Preview:")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)

        # Choose between numeric only or include categorical data
        data_option = st.radio("Choose data type:", ["Numeric Only", "Include Categorical"])

        if data_option == "Numeric Only":
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = []
        else:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Create tabs for univariate and bivariate visualizations
        univariat, bivariat = st.tabs(["📈 Univariate", "☣️ Bivariate"])

        # Univariate tabs
        tab1, tab2, tab3, tab4 = univariat.tabs(["📈 Histogram", "☣️ Box Plot", "🎌 Pie Chart", "📑 Bar Plot"])

        # Bivariate tabs (now with an extra tab for Heatmap)
        tab5, tab6, tab7, tab8 = bivariat.tabs([
            "📈 Scatterplot - 2 Columns", 
            "☣️ Bar Plot - 2 Columns", 
            "🎌 Box Plot - 2 Columns",
            "🔥 Heatmap"
        ])

        # --------- HISTOGRAM (Tab 1) ---------
        tab1.subheader("Histogram")
        if numeric_cols:
            selected_col = tab1.selectbox("Choose a column:", ["Select a column"] + numeric_cols, key="hist_col")
            if selected_col != "Select a column":
                data = df[selected_col].dropna()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
                ax.set_title(f"Distribution of {selected_col}")
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Frequency")
                tab1.pyplot(fig)
        else:
            tab1.warning("No numeric columns available for histogram visualization.")

        # --------- BOX PLOT (Tab 2) ---------
        tab2.subheader("Box Plot")
        if numeric_cols:
            selected_col = tab2.selectbox("Choose a column:", ["Select a column"] + numeric_cols, key="box_col")
            if selected_col != "Select a column":
                data = df[selected_col].dropna()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.boxplot(data, vert=False)
                ax.set_title(f"Box Plot of {selected_col}")
                ax.set_xlabel(selected_col)
                tab2.pyplot(fig)
        else:
            tab2.warning("No numeric columns available for box plot visualization.")

        # --------- PIE CHART (Tab 3) ---------
        tab3.subheader("Pie Chart")
        if categorical_cols:
            selected_col = tab3.selectbox("Choose a column:", ["Select a column"] + categorical_cols, key="pie_col")
            if selected_col != "Select a column":
                data = df[selected_col].dropna().value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.pie(data, labels=data.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
                ax.set_title(f"Pie Chart of {selected_col}")
                tab3.pyplot(fig)
        else:
            tab3.warning("No categorical columns available for pie chart visualization.")

        # --------- BAR PLOT (Tab 4) ---------
        tab4.subheader("Bar Plot")
        if numeric_cols or categorical_cols:
            selected_col = tab4.selectbox("Choose a column:", ["Select a column"] + numeric_cols + categorical_cols, key="bar_col")
            if selected_col != "Select a column":
                data = df[selected_col].dropna().value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(data.index, data.values, color='skyblue', edgecolor='black')
                ax.set_title(f"Bar Plot of {selected_col}")
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Count")
                ax.set_xticklabels(data.index, rotation=45, ha="right")
                tab4.pyplot(fig)
        else:
            tab4.warning("No columns available for bar plot visualization.")

        # --------- SCATTER PLOT (Tab 5) ---------
        tab5.subheader("Scatter Plot (2 Numeric Columns)")
        if len(numeric_cols) >= 2:
            x_col = tab5.selectbox("Choose X-axis column:", ["Select a column"] + numeric_cols, key="scatter_x")
            y_col = tab5.selectbox("Choose Y-axis column:", ["Select a column"] + numeric_cols, key="scatter_y")

            if x_col != "Select a column" and y_col != "Select a column" and x_col != y_col:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.scatter(df[x_col], df[y_col], alpha=0.7, color='blue')
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                tab5.pyplot(fig)
            else:
                tab5.warning("Please select two different numeric columns.")
        else:
            tab5.warning("Not enough numeric columns for scatter plot visualization.")

        # --------- BAR PLOT (Tab 6) ---------
        tab6.subheader("Bar Plot (1 Categorical, 1 Numeric)")
        if categorical_cols and numeric_cols:
            cat_col = tab6.selectbox("Choose Categorical Column:", ["Select a column"] + categorical_cols, key="bar_cat")
            num_col = tab6.selectbox("Choose Numeric Column:", ["Select a column"] + numeric_cols, key="bar_num")

            if cat_col != "Select a column" and num_col != "Select a column":
                grouped_data = df.groupby(cat_col)[num_col].mean().sort_values()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(grouped_data.index, grouped_data.values, color='green', edgecolor='black')
                ax.set_title(f"Bar Plot: {num_col} by {cat_col}")
                ax.set_xlabel(cat_col)
                ax.set_ylabel(num_col)
                ax.set_xticklabels(grouped_data.index, rotation=45, ha="right")
                tab6.pyplot(fig)
            else:
                tab6.warning("Please select both categorical and numeric columns.")
        else:
            tab6.warning("Not enough categorical and numeric columns for bar plot visualization.")

        # --------- BOX PLOT (Tab 7) ---------
        tab7.subheader("Box Plot (1 Categorical, 1 Numeric)")
        if categorical_cols and numeric_cols:
            cat_col = tab7.selectbox("Choose Categorical Column:", ["Select a column"] + categorical_cols, key="box_cat")
            num_col = tab7.selectbox("Choose Numeric Column:", ["Select a column"] + numeric_cols, key="box_num")

            if cat_col != "Select a column" and num_col != "Select a column":
                data = df[cat_col].dropna().value_counts()
                fig, ax = plt.subplots(figsize=(8, 5))
                df.boxplot(column=num_col, by=cat_col, ax=ax)
                ax.set_title(f"Box Plot: {num_col} by {cat_col}")
                ax.set_xlabel(cat_col)
                ax.set_ylabel(num_col)
                ax.set_xticklabels(data.index, rotation=45, ha="right")
                tab7.pyplot(fig)
            else:
                tab7.warning("Please select both categorical and numeric columns.")
        else:
            tab7.warning("Not enough categorical and numeric columns for box plot visualization.")

        # --------- HEATMAP (Tab 8) ---------
        tab8.subheader("Correlation Heatmap")
        if len(numeric_cols) >= 2:
            # Compute correlation matrix using only the numeric columns
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Correlation Heatmap")
            tab8.pyplot(fig)
        else:
            tab8.warning("Not enough numeric columns to compute a correlation matrix.")
