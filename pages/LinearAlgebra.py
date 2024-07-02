import streamlit as st

class LinearAlgebra:
    def __init__(self):
        pass
    def includes(self):
        st.write("Includes:")
        st.write("1. Vectors")
        st.write("----> 1.1. What is a vector?")
        st.write("----> 1.2. Vector operations")
        st.write("----> 1.3. Basis vectors")
        st.write("----> 1.4. Dot or inner product")
        st.write("----> 1.5. Scalar and Vector projection")
        st.write("2. Matrices")
        st.write("----> 2.1. What is a matrix?")
        st.write("----> 2.2. Matrix operations")
        st.write("----> 2.3. Identity matrix")
        st.write("----> 2.4. Determinants of a Matrix")
        st.write("----> 2.5. Inverse of a Matrix")
        st.write("----> 2.6. Special Matrices")
        st.write("6. Change of basis")
        st.write("----> 6.1. Transformation Matrix")
        st.write("----> 6.2. Orthonormal Basis")
        st.write("----> 6.3. Gram-Schmidt Process")
        st.write("----> 6.4. Transformations in a Plane")
        st.write("9. Eigenvalues and Eigenvectors (Eigenstuff)")
        st.write("----> 9.1. Definition and Basic Concepts")
        st.write("----> 9.2. Finding Eigenvalues and Eigenvectors")
        st.write("----> 9.3. Properties of Eigenvalues and Eigenvectors")
        st.write("----> 9.4. Applications")
        st.write("----> 9.5. Power Method")

    def introductionToLinearAlgebra(self):
        st.write("## Introduction to Linear Algebra")
        st.write("Linear algebra is a branch of mathematics that is widely used throughout science and engineering. It is used to solve systems of linear equations, compute simple matrix decompositions, and perform many other types of calculations. In this section, we will introduce the basic concepts of linear algebra, including vectors, matrices, determinants, and eigenvalues and eigenvectors.")
        st.write("Here are some of the topics that we will cover in this section:")
        self.includes()

    def vectors(self):
        st.markdown("## Vectors")
        st.latex(r"""\textbf{A \ vector\ is\ a\ mathematical\ object\ that\ has\ both\ magnitude\ and\ direction.\ Vectors\ are\ often\ used\ to\ represent\ physical\ quantities\ such\ as\ velocity, force, and electric field. In this section, we will introduce the basic concepts of vectors, \
                  including vector operations, basis vectors, dot or inner product, and scalar and vector projection.}""")
        st.markdown(r"Here are some of the topics that we will cover in this section:")

        st.markdown("### 1. What is a vector?")
        st.latex(r"   - A\ vector\ is\ a\ quantity\ that\ has\ both\ magnitude\ (or\ length)\ and\ direction.\ It\ is\ often\ represented\ graphically\ by\ an\ arrow,\ where\ the\ length\ of\ the\ arrow\ denotes\ the\ magnitude\ and\ the\ direction\ of\ the\ arrow\ indicates\ the\ direction.")
        st.latex(r"   - In\ mathematical\ notation,\ a\ vector\ is\ typically\ written\ in\ boldface\ (e.g.,\ \mathbf{v})\ or\ with\ an\ arrow\ over\ it\ (e.g.,\ \vec{v}).\ Vectors\ can\ exist\ in\ any\ number\ of\ dimensions,\ but\ in\ most\ applications,\ we\ deal\ with\ 2D\ or\ 3D\ vectors.")

        st.markdown("### 2. Vector operations")
        st.latex(r"   - \textbf{Addition\ and\ Subtraction}:\ Vectors\ can\ be\ added\ or\ subtracted\ by\ adding\ or\ subtracting\ their\ corresponding\ components.\ For\ example,\ if\ \mathbf{a}\ =\ (a1,\ a2)\ and\ \mathbf{b}\ =\ (b1,\ b2),\ then\ \mathbf{a}\ +\ \mathbf{b}\ =\ (a1\ +\ b1,\ a2\ +\ b2).")
        st.latex(r"   - \textbf{Scalar\ Multiplication}:\ A\ vector\ can\ be\ multiplied\ by\ a\ scalar\ (a\ real\ number)\ by\ multiplying\ each\ component\ of\ the\ vector\ by\ the\ scalar.\ For\ example,\ if\ \mathbf{v}\ =\ (v1,\ v2)\ and\ k\ is\ a\ scalar,\ then\ k\mathbf{v}\ =\ (kv1,\ kv2).")
        st.latex(r"   - \textbf{Magnitude}:\ The\ magnitude\ of\ a\ vector\ \mathbf{v}\ =\ (v1,\ v2)\ is\ given\ by\ \| \vec{v}\ \|\ =\ \sqrt{v1^2\ +\ v2^2}.\ In\ 3D,\ for\ \mathbf{v}\ =\ (v1,\ v2,\ v3),\ the\ magnitude\ is\ \| \vec{v}\ \|\ =\ \sqrt{v1^2\ +\ v2^2\ +\ v3^2}.")
        st.latex(r"   - \textbf{Dot\ Product}:\ The\ dot\ product\ (or\ inner\ product)\ of\ two\ vectors\ \mathbf{a}\ and\ \mathbf{b}\ is\ given\ by\ \mathbf{a}\ \cdot\ \mathbf{b}\ =\ a1b1\ +\ a2b2\ in\ 2D,\ or\ \mathbf{a}\ \cdot\ \mathbf{b}\ =\ a1b1\ +\ a2b2\ +\ a3b3\ in\ 3D.\ This\ product\ is\ commutative\ and\ distributive\ over\ vector\ addition.")
        st.latex(r"   - \textbf{Properties\ of\ Dot\ Product}:\ The\ dot\ product\ is\ commutative\ (\mathbf{a}\ \cdot\ \mathbf{b}\ =\ \mathbf{b}\ \cdot\ \mathbf{a}),\ distributive\ over\ vector\ addition\ (\mathbf{a}\ \cdot\ (\mathbf{b}\ +\ \mathbf{c})\ =\ \mathbf{a}\ \cdot\ \mathbf{b}\ +\ \mathbf{a}\ \cdot\ \mathbf{c}),\ and\ associative\ with\ scalar\ multiplication\ (k(\mathbf{a}\ \cdot\ \mathbf{b})\ =\ (k\mathbf{a})\ \cdot\ \mathbf{b}\ =\ \mathbf{a}\ \cdot\ (k\mathbf{b})).")

        st.markdown("### 3. Basis vectors")
        st.latex(r"   - Basis\ vectors\ are\ a\ set\ of\ vectors\ that\ are\ used\ to\ define\ a\ vector\ space.\ In\ 2D,\ the\ standard\ basis\ vectors\ are\ \mathbf{i}\ =\ (1,\ 0)\ and\ \mathbf{j}\ =\ (0,\ 1).\ In\ 3D,\ the\ standard\ basis\ vectors\ are\ \mathbf{i}\ =\ (1,\ 0,\ 0),\ \mathbf{j}\ =\ (0,\ 1,\ 0),\ and\ \mathbf{k}\ =\ (0,\ 0,\ 1).")
        st.latex(r"   - Any\ vector\ in\ the\ space\ can\ be\ expressed\ as\ a\ linear\ combination\ of\ the\ basis\ vectors.\ For\ example,\ in\ 2D,\ a\ vector\ \mathbf{v}\ =\ (v1,\ v2)\ can\ be\ written\ as\ \mathbf{v}\ =\ v1\mathbf{i}\ +\ v2\mathbf{j}.")
        st.latex(r"   - A\ basis\ is\ a\ set\ of\ n\ vectors\ that:\ (i)\ are\ not\ linear\ combinations\ of\ each\ other,\ and\ (ii)\ span\ the\ space.\ The\ space\ is\ then\ n-dimensional.")

        st.markdown("### 4. Dot or inner product")
        st.latex(r"   - The\ dot\ product\ (also\ known\ as\ the\ inner\ product)\ of\ two\ vectors\ \mathbf{a}\ and\ \mathbf{b}\ is\ a\ scalar\ value\ that\ is\ calculated\ as\ \mathbf{a}\ \cdot\ \mathbf{b}\ =\ a1b1\ +\ a2b2\ in\ 2D\ or\ \mathbf{a}\ \cdot\ \mathbf{b}\ =\ a1b1\ +\ a2b2\ +\ a3b3\ in\ 3D.")
        st.latex(r"   - The\ dot\ product\ is\ used\ to\ determine\ the\ angle\ between\ two\ vectors\ and\ to\ test\ for\ orthogonality.\ If\ the\ dot\ product\ is\ zero,\ the\ vectors\ are\ perpendicular.")
        st.latex(r"   - The\ dot\ product\ also\ appears\ in\ the\ formula\ for\ the\ projection\ of\ one\ vector\ onto\ another\ and\ in\ calculating\ work\ done\ by\ a\ force.")
        st.latex(r"   - Properties:\ The\ dot\ product\ is\ commutative\ (\mathbf{a}\ \cdot\ \mathbf{b}\ =\ \mathbf{b}\ \cdot\ \mathbf{a}),\ distributive\ (\mathbf{a}\ \cdot\ (\mathbf{b}\ +\ \mathbf{c})\ =\ \mathbf{a}\ \cdot\ \mathbf{b}\ +\ \mathbf{a}\ \cdot\ \mathbf{c}),\ and\ associative\ with\ scalar\ multiplication\ (k(\mathbf{a}\ \cdot\ \mathbf{b})\ =\ (k\mathbf{a})\ \cdot\ \mathbf{b}\ =\ \mathbf{a}\ \cdot\ (k\mathbf{b})).")

        st.markdown("### 5. Scalar and Vector projection")
        st.latex(r"- \textbf{Scalar \ Projection The scalar projection of a vector  $\mathbf{a}$ onto a vector $\mathbf{b}$ is given by the formula $ \text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\| \mathbf{b} \\|}$. It represents the length of the projection of $\mathbf{a}$ onto $\mathbf{b}$.}")
        st.latex(r"- \textbf{Vector Projection The vector projection of $\mathbf{a}$ onto $\mathbf{b}$ is given by $\text{Proj}_{\mathbf{b}} \mathbf{a} = \left(\frac{\mathbf{a} \cdot \mathbf{b}}{\| \mathbf{b} \|^2}\right) \mathbf{b}$. It represents the vector that points in the direction of $\mathbf{b}$ and whose length is the scalar projection of $\mathbf{a}$ onto $\mathbf{b}$.}",)
        st.latex(r"- \textbf{Scalar projection formula $\text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\| \mathbf{b} \|}$}")
        st.latex(r"- \textbf{Vector projection formula $\text{Proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\| \mathbf{b} \|^2} \mathbf{b}$}")



    def matrices(self):
        st.write("## Matrices")
        st.latex(r"- \textbf{Matrices are rectangular arrays of numbers that are used to represent linear transformations and systems of linear equations. In this section, we will introduce the basic concepts of matrices, including matrix operations, identity matrix, determinants, and matrix inversion.}")
        st.latex(r"- \textbf{Here are some of the topics that we will cover in this section:}")

        st.write("### 1. Matrix Operations")
        st.latex(r"""- \textbf{Matrix Addition and Subtraction: Matrices of the same dimension can be added or subtracted by adding or subtracting their corresponding elements. For example, if \( A \) and \( B \) are two matrices of the same size, their sum \( C = A + B \) is given by \( C_{ij} = A_{ij} + B_{ij} \).}""")
        st.latex(r"""- \textbf{Scalar Multiplication: A matrix can be multiplied by a scalar (a real number) by multiplying each element of the matrix by the scalar. For example, if \( A \) is a matrix and \( k \) is a scalar, then \( kA \) is a matrix whose elements are \( (kA)_{ij} = kA_{ij} \).}""")
        st.latex(r"""- \textbf{Matrix Multiplication: The product of two matrices \( A \) and \( B \) is a new matrix \( C \) whose elements are given by \( C_{ij} = \sum_k A_{ik} B_{kj} \). This operation is not commutative, meaning \( AB \neq BA \) in general.}""")

        # Identity Matrix
        st.write("### 2. Identity Matrix")
        st.latex(r"""
        - \textbf{The identity matrix, denoted by \( I \), is a square matrix with ones on the diagonal and zeros elsewhere. For example, the 2x2 identity matrix is \( I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \). Multiplying any matrix by the identity matrix leaves the original matrix unchanged, i.e., \( AI = IA = A \).}
        """)

        st.write("### 3. Determinants of a Matrix")
        st.latex(r"""
        - \textbf{The determinant is a scalar value that can be computed from the elements of a square matrix. It provides important properties of the matrix, such as whether it is invertible.}
        """)
        st.latex(r"""
        - \textbf{Determinant of a 2x2 Matrix: For a 2x2 matrix \( A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \), the determinant is given by \( \det(A) = ad - bc \)}.
        """)
        st.latex(r"""
        - \textbf{Determinant of a 3x3 Matrix: For a 3x3 matrix \( A = \begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} \), the determinant is given by \( \det(A) = a(ei - fh) - b(di - fg) + c(dh - eg) \).}
        """)
        st.write("### 4. Inverse of a Matrix")
        st.latex(r"""
        - \textbf{The inverse of a matrix \( A \) is a matrix \( A^{-1} \) such that \( AA^{-1} = A^{-1}A = I \). For a 2x2 matrix \( A = \begin{pmatrix} a & b \\ c & d \end{pmatrix} \), if \( ad - bc \neq 0 \), the inverse is given by \( A^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} \).}
        """)
        st.latex(r"""
        - \textbf{For larger matrices, finding the inverse involves more complex methods such as Gaussian elimination or using the adjugate matrix and the determinant.}
        """)

        st.write("### 5. Special Matrices")
        st.latex(r"""
        - \textbf{Orthonormal Matrix: A matrix is orthonormal if its columns are unit vectors and orthogonal to each other. For an orthonormal matrix \( A \), \( A^T = A^{-1} \).}
        """)
        st.latex(r"""
        - \textbf{Rotation Matrix: A matrix that represents a rotation transformation in a plane is given by \( \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix} \), where \( \theta \) is the angle of rotation.}
        """)
    def changeOfBasis(self):
        st.write("## Change of Basis")
        st.latex(r"""
        \textbf{Changing the basis of a vector space involves transforming vectors expressed in one basis to another basis. This concept is crucial in many applications, including simplifying matrix operations and solving linear systems.}
        """)
        st.latex(r"""
        \textbf{Here are some of the topics that we will cover in this section:}
        """)

        st.write("### 1. Transformation Matrix")
        st.latex(r"""
        \text{The columns of the transformation matrix } B \text{ are the new basis vectors expressed in the original coordinate system. For a vector } r \text{ in the original basis and } r' \text{ in the new basis, the relationship is given by:}
        """)
        st.latex(r"""
        Br' = r
        """)
        st.latex(r"""
        \text{or equivalently,}
        """)
        st.latex(r"""
        r' = B^{-1} r
        """)
        st.write("### 2. Orthonormal Basis")
        st.latex(r"""
        - \textbf{If a matrix \( A \) is orthonormal, meaning all its columns are unit vectors and orthogonal to each other, then the transpose of \( A \) is equal to its inverse:}
        """)
        st.latex(r"""
        A^T = A^{-1}
        """)

        st.write("### 3. Gram-Schmidt Process")
        st.latex(r"""
        - \textbf {The Gram-Schmidt process is used to construct an orthonormal basis from a set of linearly independent vectors. Starting with \( n \) linearly independent basis vectors \( \{v_1, v_2, ..., v_n\} \), the process is as follows:}""")
        st.latex(r"""- \textbf{Normalize the first vector:}""")
        st.latex(r"""
        e_1 = \frac{v_1}{||v_1||}
        """)
        st.write(r"""
            - For each subsequent vector, subtract the projection of the vector onto all previously obtained orthonormal vectors and then normalize:
        """)
        st.latex(r"""
        u_2 = v_2 - (v_2 \cdot e_1)e_1, \quad e_2 = \frac{u_2}{||u_2||}
        """)
        st.latex(r"""
        u_3 = v_3 - (v_3 \cdot e_1)e_1 - (v_3 \cdot e_2)e_2, \quad e_3 = \frac{u_3}{||u_3||}
        """)
        st.write(r"""
        - Continue this process for all vectors to obtain an orthonormal set.
        """)

        st.write("### 4. Transformations in a Plane")
        st.write(r"""
        - To transform vectors or objects in a plane, first transform them into the basis referred to the reflection plane (or other transformation plane), perform the transformation, and then transform back into the original basis:
        """)
        st.latex(r"""
        r' = ETE^{-1}r
        """)
        st.write(r"""
        where \( E \) is the matrix representing the basis transformation and \( T \) is the transformation in the new basis.
        """)
    def eigenvaluesAndEigenvectors(self):
        st.write("## Eigenvalues and Eigenvectors")
        st.write(r"""
        Eigenvalues and eigenvectors are fundamental in understanding linear transformations and their effects on vector spaces. They have various applications in different fields, including machine learning, physics, and engineering.
        """)

        st.write("### 1. Definition and Basic Concepts")
        st.latex(r"""
        - \textbf{Eigenvalues: For a given square matrix \( A \), an eigenvalue \( \lambda \) is a scalar such that there exists a non-zero vector \( \mathbf{x} \) (called the eigenvector) satisfying:}
        """)
        st.latex(r"""
        A\mathbf{x} = \lambda \mathbf{x}
        """)
        st.latex(r"""
        - \textbf{Eigenvectors: The vector \( \mathbf{x} \) corresponding to the eigenvalue \( \lambda \) is called an eigenvector of \( A \).
        - Eigenvalue Equation: Rearranging the above equation, we get:}
        """)
        st.latex(r"""
        (A - \lambda I)\mathbf{x} = 0
        """)
        st.latex(r"""
        \textbf{where \( I \) is the identity matrix. For non-trivial solutions, the determinant must be zero:}
        """)
        st.latex(r"""
        \det(A - \lambda I) = 0
        """)

        st.write("### 2. Finding Eigenvalues and Eigenvectors")
        st.latex(r"""
        - \textbf{Step-by-Step Process:}""")
        st.latex(r"""
        \textbf{1. Characteristic Polynomial: Calculate the characteristic polynomial \( p(\lambda) = \det(A - \lambda I) \).}
        """)
        st.latex(r"""\textbf{2. Solve for Eigenvalues: Find the roots of the characteristic polynomial. These roots are the eigenvalues.}""")
        st.latex(r"""\textbf{3. Find Eigenvectors: For each eigenvalue \( \lambda \), solve the linear system \( (A - \lambda I)\mathbf{x} = 0 \) to find the corresponding eigenvector.}
        """)

        st.write("#### Example")
        st.latex(r"""
        \textbf{Consider the matrix \( A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix} \).}""")
        st.latex(r"""\textbf{1. Characteristic Polynomial:}""")
        st.latex(r"""
        \det(A - \lambda I) = \det\begin{pmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{pmatrix} = (4-\lambda)(3-\lambda) - 2 \cdot 1
        """)
        st.latex(r"""\
        = \lambda^2 - 7\lambda + 10 - 2 = \lambda^2 - 7\lambda + 8
        """)
        st.write(r"""
        2. **Solve for Eigenvalues**:
        """)
        st.latex(r"""
        \lambda^2 - 7\lambda + 8 = 0
        """)
        st.latex(r"""
        \textbf{Solving this quadratic equation gives the eigenvalues \( \lambda_1 = 4 \) and \( \lambda_2 = 3 \)}.""")
        st.latex(r"""\textbf{3. Find Eigenvectors:}""")
        st.latex(r"""\textbf{- For \( \lambda_1 = 4 \)}:""")
        st.latex(r"""
        (A - 4I)\mathbf{x} = \begin{pmatrix} 0 & 1 \\ 2 & -1 \end{pmatrix}\mathbf{x} = 0
        """)
        st.latex(r"""\textbf{
        Solving this system gives the eigenvector \( \mathbf{x}_1 = \begin{pmatrix} 1 \\ 2 \end{pmatrix} \).}""")
        st.latex(r"""-\textbf{ For \( \lambda_2 = 3 \):}""")
        st.latex(r"""
        (A - 3I)\mathbf{x} = \begin{pmatrix} 1 & 1 \\ 2 & 0 \end{pmatrix}\mathbf{x} = 0
        """)
        st.latex(r"""
        \textbf{Solving this system gives the eigenvector \( \mathbf{x}_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix} \).}
        """)

        st.write("### 3. Properties of Eigenvalues and Eigenvectors")
        st.latex(r"""
        \textbf{- Linearity: If \( \mathbf{x} \) is an eigenvector of \( A \) corresponding to the eigenvalue \( \lambda \), then any scalar multiple of \( \mathbf{x} \) is also an eigenvector corresponding to \( \lambda \).}""")
        st.latex(r"""
        \textbf{- Sum of Eigenvalues: The sum of the eigenvalues of a matrix is equal to the trace of the matrix (the sum of its diagonal elements).}""")
        st.latex(r"""- \textbf{Product of Eigenvalues: The product of the eigenvalues of a matrix is equal to the determinant of the matrix.}""")
        st.latex(r"""- \textbf{Orthogonality: Eigenvectors corresponding to distinct eigenvalues of a symmetric matrix are orthogonal.}""")

        st.write("### 4. Applications")
        st.write(r"""
        - **Stability Analysis**: Eigenvalues are used to analyze the stability of equilibrium points in dynamical systems.
        - **Vibration Analysis**: In mechanical systems, eigenvalues and eigenvectors are used to determine natural frequencies and mode shapes.
        - **Principal Component Analysis (PCA)**: In statistics, PCA uses eigenvalues and eigenvectors of the covariance matrix to reduce the dimensionality of data.
        - **Google's PageRank**: The PageRank algorithm uses the dominant eigenvector of the link matrix to rank web pages. The Power Method is often used to find this dominant eigenvector.
        - **Quantum Mechanics**: Eigenvalues and eigenvectors are fundamental in solving the Schrödinger equation and understanding the energy levels of quantum systems.
        """)

        st.write("### 5. Power Method")
        st.latex(r"""
        - \textbf{Iterative Algorithm: The Power Method is an iterative algorithm used to find the dominant eigenvalue and its corresponding eigenvector. Starting from an initial vector \( \mathbf{r}_0 \), the method iteratively applies the matrix:}
        """)
        st.latex(r"""
        \mathbf{r}_{k+1} = \frac{A\mathbf{r}_k}{||A\mathbf{r}_k||}
        """)
        st.latex(r"""
        \textbf{The vector \( \mathbf{r}_k \) converges to the dominant eigenvector of \( A \).}
        """)

        st.write("### Example: PageRank")
        st.write(r"""
        Consider a link matrix \( L \) for a web with three pages:
        """)
        st.latex(r"""
        L = \begin{pmatrix} 0 & 1 & 1 \\ 1 & 0 & 0 \\ 1 & 1 & 0 \end{pmatrix}
        """)
        st.latex(r"""
        \textbf{Using the Power Method with a damping factor \( d \), the PageRank vector \( \mathbf{r} \) is calculated as:}
        """)
        st.latex(r"""
        \mathbf{r}_{k+1} = dL\mathbf{r}_k + \frac{1-d}{n} \mathbf{1}
        """)
        st.latex(r"""
        \textbf{where \( \mathbf{1} \) is a vector of ones, and \( n \) is the number of pages.}
        """)
        st.write("### References")
        st.write(r"""
        This section is based on the material from the provided Maths For Machine Learning specialization on Coursera.
                 Thanks to the instructors and the Coursera platform for providing this valuable content.
        """)

    def run(self):
        st.title("Linear Algebra :triangular_ruler:")
        self.introductionToLinearAlgebra()
        self.vectors()
        self.matrices()
        self.changeOfBasis()
        self.eigenvaluesAndEigenvectors()

def main():
    st.set_page_config(page_title="Linear Algebra", page_icon=":triangular_ruler:", layout="wide")
    linearAlgebra = LinearAlgebra()
    linearAlgebra.run()

if __name__ == "__main__":
    main()