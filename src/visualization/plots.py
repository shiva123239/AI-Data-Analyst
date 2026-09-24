import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Common chart settings
# ---------------------------------------------------------

CHART_SIZE = (7, 3.5)


# ---------------------------------------------------------
# Histogram
# ---------------------------------------------------------

def plot_histogram(df, column):
    """
    Create a compact histogram for a numeric column.
    """

    fig, ax = plt.subplots(figsize=CHART_SIZE)

    sns.histplot(
        data=df,
        x=column,
        kde=True,
        ax=ax
    )

    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")

    plt.tight_layout()

    return fig


# ---------------------------------------------------------
# Boxplot
# ---------------------------------------------------------

def plot_boxplot(df, column):
    """
    Create a compact boxplot for a numeric column.
    """

    fig, ax = plt.subplots(figsize=CHART_SIZE)

    sns.boxplot(
        data=df,
        y=column,
        ax=ax
    )

    ax.set_title(f"Boxplot of {column}")
    ax.set_ylabel(column)

    plt.tight_layout()

    return fig


# ---------------------------------------------------------
# Category Bar Chart
# ---------------------------------------------------------

def plot_bar(df, column):
    """
    Create a compact bar chart for a categorical column.

    Only the top 10 most frequent categories are displayed.
    """

    fig, ax = plt.subplots(figsize=CHART_SIZE)

    counts = (
        df[column]
        .astype(str)
        .value_counts()
        .head(10)
    )

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(f"Top 10 {column} Categories")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    return fig


# ---------------------------------------------------------
# Category Bar Chart - alternate function name
# ---------------------------------------------------------

def plot_category_bar(df, column):
    """
    Alias for the category bar chart.
    """

    return plot_bar(df, column)


# ---------------------------------------------------------
# Category Bar Chart - alternate function name
# ---------------------------------------------------------

def plot_category_bar_chart(df, column):
    """
    Alias for the category bar chart.
    """

    return plot_bar(df, column)


# ---------------------------------------------------------
# Scatter Plot
# ---------------------------------------------------------

def plot_scatter(df, x_column, y_column):
    """
    Create a compact scatter plot between two numeric columns.
    """

    fig, ax = plt.subplots(figsize=CHART_SIZE)

    sns.scatterplot(
        data=df,
        x=x_column,
        y=y_column,
        ax=ax
    )

    ax.set_title(f"{y_column} vs {x_column}")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)

    plt.tight_layout()

    return fig


# ---------------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------------

def plot_correlation(df):
    """
    Create a compact correlation heatmap
    for numeric columns.
    """

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))

    correlation = numeric_df.corr()

    sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    ax.set_title("Correlation Matrix")

    plt.tight_layout()

    return fig


# ---------------------------------------------------------
# Correlation Heatmap - alternate function name
# ---------------------------------------------------------

def plot_correlation_heatmap(df):
    """
    Alias for the correlation heatmap.
    """

    return plot_correlation(df)