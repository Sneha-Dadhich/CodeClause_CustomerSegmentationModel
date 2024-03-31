import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import streamlit as st
import sklearn

    # This program uses new syntaxes of panda3 at certain points

# importing the file
path = 'C:\\Sneha\\Programs1\\Python\\Internship\\CodeClause\\CustomerSegmenttation\\Mall_Customers.csv'
fields = ("CustomerID", "Gender", "Age", "Annual Income(k$)", "Spending Score(1-100)")
raw_data = pd.read_csv(path, names=fields, header=0)
data = pd.DataFrame(raw_data, columns=fields)

def categorical_gender(data):
    """If the gender column will be a categorical, the function will transform it into numerical.
    Returns the updated dataframe"""
    data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
    return data

def age_vs_spending_score(data):
    """Creates a scatter plot between Age and Spending score of a customer
    returns a scatter plot
    """
    plt.figure(figsize=(10, 6))
    data.plot(kind='scatter', x='Age', y='Spending Score(1-100)', marker='o', ax=plt.gca())
    plt.xlabel('Age')
    plt.ylabel('Spending Score')
    plt.title('Scatter plot between Age and Spending Score')
    return plt

def age_vs_annualincome(data):
    """Creates a scatter plot between age and annual income
    returns a scatter plot
    """
    plt.figure(figsize=(10,6))
    data.plot(kind="scatter",x="Age",y="Annual Income(k$)",marker="o",ax=plt.gca())
    plt.xlabel('Age')
    plt.ylabel('Annual Income')
    plt.title('Scatter plot between Age and Annual Income')
    return plt

def annualincome_vs_spendingscore(data):
    """Creates a scatter plot between income and spending score
    Returns a scatter plot
    """
    plt.figure(figsize=(10,6))
    data.plot(kind="scatter",x="Annual Income(k$)",y="Spending Score(1-100)",marker="o",ax=plt.gca());
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score')
    plt.title('Scatter plot between Annual Income and Spending Score')
    return plt

def gender_vs_spendingscore(data):
    """Creates a scatter plot between gender and spending score
    Returns a scatter plot
    """
    plt.figure(figsize=(10,6))
    sns.violinplot(x='Gender', y='Spending Score(1-100)', data=data)
    plt.title('Distribution of Spending Score by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Spending Score')
    return plt

def corr(data):
    """Plots a correlation heatmap between variables
    Returns a correlation heatmap"""
    data = categorical_gender(data)
    
    fig_dims = (6, 6)  # reduced plot size
    fig, ax = plt.subplots(figsize=fig_dims)  # added ax to capture axes
    sns.heatmap(data.corr(), annot=True, cmap='viridis', ax=ax)  # removed extra corr() call
    return fig  # returning fig instead of sns

data = categorical_gender(data)    
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Age', "Annual Income(k$)", "Spending Score(1-100)"]])


def elbow(data):
    """Returns optimal number of clusters data should be divided to
    Returns a plot"""
    wcss = []
    for i in range(1, 11):
        kmeans = sklearn.cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        x = data.copy()
        kmeans.fit(x)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)

    plt.figure(figsize=(10, 5))
    no_clusters = range(1, 11)
    plt.plot(no_clusters, wcss, marker="o")
    plt.title('The elbow method', fontweight="bold")
    plt.xlabel('Number of clusters(K)')
    plt.ylabel('within Clusters Sum of Squares(WCSS)')
    return plt

def clustering_new(data):
    """Returns mall customers divided into 5 clusters based on annual income and spending score
    Returns a scatter plot"""

    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)

    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    gender= {0:'Male', 1:'Female'}  # Assuming 0 corresponds to Male and 1 corresponds to Female
    clusters_new['Gender'] = clusters_new['Gender'].map(gender)
    
    plt.figure(figsize=(6,6))
    # Using 'c' argument instead of 'color' for scatter plot
    data.plot(x='Annual Income(k$)', y='Spending Score(1-100)', c=clusters_new['cluster_pred'], cmap='rainbow', kind='scatter')
    plt.title("Clustering customers based on Annual Income and Spending score", fontsize=15, fontweight="bold")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    return plt


def barplot_age(data):
    """"Visualizes clusters based on age
    Returns a bar plot"""
    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)
    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    avg_data = clusters_new.groupby(['cluster_pred'], as_index=False).mean()
    sns.barplot(x='cluster_pred',y='Age',palette="plasma",data=avg_data)
    plt.savefig("barplot_age.png")
    return plt.gcf()

def barplot_annualincome(data):
    """"Visualizes clusters based on Annual Income
        Returns a bar plot"""
    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)
    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    avg_data = clusters_new.groupby(['cluster_pred'], as_index=False).mean()
    sns.barplot(x='cluster_pred', y='Annual Income(k$)', palette="plasma", data=avg_data)
    plt.savefig("barplot_annualincome.png")
    return plt.gcf()

def barplot_spendingscore(data):
    """"Visualizes clusters based on Spending Scores
        Returns a bar plot"""
    x = data.copy()
    kmeans_new = KMeans(5)
    kmeans_new.fit(x)
    clusters_new = x.copy()
    clusters_new['cluster_pred'] = kmeans_new.fit_predict(x)
    avg_data = clusters_new.groupby(['cluster_pred'], as_index=False).mean()
    sns.barplot(x='cluster_pred',y='Spending Score(1-100)',palette="plasma",data=avg_data)
    plt.savefig("bar_spendingScore.png")
    return plt.gcf()

def Mera_KMeans(data):
    """Performs KMeans clustering and visualizes the results
    Returns a plot"""
    km = KMeans(n_clusters=5)
    clusters = km.fit_predict(data.iloc[:,1:])
    data["label"] = clusters
 
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.Age[data.label == 0], data["Annual Income(k$)"][data.label == 0], data["Spending Score(1-100)"][data.label == 0], c='blue', s=60)
    ax.scatter(data.Age[data.label == 1], data["Annual Income(k$)"][data.label == 1], data["Spending Score(1-100)"][data.label == 1], c='red', s=60)
    ax.scatter(data.Age[data.label == 2], data["Annual Income(k$)"][data.label == 2], data["Spending Score(1-100)"][data.label == 2], c='green', s=60)
    ax.scatter(data.Age[data.label == 3], data["Annual Income(k$)"][data.label == 3], data["Spending Score(1-100)"][data.label == 3], c='orange', s=60)
    ax.scatter(data.Age[data.label == 4], data["Annual Income(k$)"][data.label == 4], data["Spending Score(1-100)"][data.label == 4], c='purple', s=60)
    ax.view_init(30, 185)
    plt.xlabel("Age")
    plt.ylabel("Annual Income (k$)")
    ax.set_zlabel('Spending Score (1-100)')
    return plt

def main():
    age_vs_spending_score(data)
    age_vs_annualincome(data).show()
    annualincome_vs_spendingscore(data)
    gender_vs_spendingscore(data)
    corr(data)
    plt.show()
    elbow(data)
    clustering_new(data)
    barplot_age(data)
    barplot_annualincome(data)
    barplot_spendingscore(data)
    Mera_KMeans(data)
    
    plt.show()


# Base UI for above model
# Define background color and page width
PAGE_BG_COLOR = "#f0f0f0"
PAGE_WIDTH = 1200

# Define a custom CSS style to center align the charts
def set_custom_css():
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {PAGE_WIDTH}px;
                padding-top: 2rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 2rem;
            }}
            .reportview-container .main {{
                color: #333;
                background-color: {PAGE_BG_COLOR};
            }}
            .css-1eh1jz6 {{
                margin-top: 3rem;
                margin-bottom: 3rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Base UI for above model
import streamlit as st
import matplotlib.pyplot as plt

# Define background color and page width
PAGE_BG_COLOR = "#f0f0f0"
PAGE_WIDTH = 1200

# Define a custom CSS style to center align the charts
def set_custom_css():
    st.markdown(
        f"""
        <style>
            .reportview-container .main .block-container {{
                max-width: {PAGE_WIDTH}px;
                padding-top: 2rem;
                padding-right: 1rem;
                padding-left: 1rem;
                padding-bottom: 2rem;
            }}
            .reportview-container .main {{
                color: #333;
                background-color: {PAGE_BG_COLOR};
            }}
            .css-1eh1jz6 {{
                margin-top: 3rem;
                margin-bottom: 3rem;
            }}
            .sidebar .sidebar-content {{
                padding: 2rem;
            }}
            .sidebar .sidebar-content .sidebar-section {{
                margin-bottom: 2rem;
            }}
            .sidebar .sidebar-content .sidebar-section:first-child {{
                margin-top: 2rem;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Base UI for above model
def main():
    set_custom_css()
    st.title('Customer Data Analysis')
    st.header("Customer Analysis of Mall" )
    # Sidebar for buttons
    st.sidebar.header('Options')

    data = None  # Load your data from the uploaded file here
if st.sidebar.button('Age vs. Spending Score'):
    fig = age_vs_spending_score(data)
    st.pyplot(fig)

if st.sidebar.button('Age vs. Annual Income'):
    fig = age_vs_annualincome(data)
    st.pyplot(fig)

if st.sidebar.button('Annual Income vs. Spending Score'):
    fig = annualincome_vs_spendingscore(data)
    st.pyplot(fig)

if st.sidebar.button('Gender vs. Spending Score'):
    fig = gender_vs_spendingscore(data)
    st.pyplot(fig)

if st.sidebar.button('Correlation Heatmap'):
    fig = corr(data)
    st.pyplot(fig)

if st.sidebar.button('Elbow Method for KMeans'):
    fig = elbow(data)
    st.pyplot(fig)

if st.sidebar.button('Perform KMeans Clustering'):
    fig = clustering_new(data)
    st.pyplot(fig)

if st.sidebar.button('Bar Plot of Age'):
    fig = barplot_age(data)
    st.pyplot(fig)

if st.sidebar.button('Bar Plot of Annual Income'):
    fig = barplot_annualincome(data)
    st.pyplot(fig)

if st.sidebar.button('Bar Plot of Spending Score'):
    fig = barplot_spendingscore(data)
    st.pyplot(fig)

if st.sidebar.button('Custom KMeans Function'):
    fig = Mera_KMeans(data)
    st.pyplot(fig)

if __name__ == '__main__':
    main()

'''
if __name__ == '__main__':
    main()

    if st.sidebar.button('Correlation Heatmap'):
        fig = corr(data)
        st.pyplot(fig)

    if st.sidebar.button('Elbow Method for KMeans'):
        fig = elbow(data)
        st.pyplot(fig)

    if st.sidebar.button('Perform KMeans Clustering'):
        fig = clustering_new(data)
        st.pyplot(fig)

    if st.sidebar.button('Bar Plot of Age'):
        fig = barplot_age(data)
        st.pyplot(fig)

    if st.sidebar.button('Bar Plot of Annual Income'):
        fig = barplot_annualincome(data)
        st.pyplot(fig)

    if st.sidebar.button('Bar Plot of Spending Score'):
        fig = barplot_spendingscore(data)
        st.pyplot(fig)

    if st.sidebar.button('Custom KMeans Function'):
        fig = Mera_KMeans(data)
        st.pyplot(fig)
 
if __name__ == '__main__':
    main()
'''