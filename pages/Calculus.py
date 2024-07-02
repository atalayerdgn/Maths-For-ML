import streamlit as st

class CalculusApp:
    def __init__(self):
        pass

    def introduction(self):
        st.title("Multivariate Calculus")
        st.latex(r''' \text{Calculus is the study of how things change.} ''')
    def includes(self):
        st.write("Includes:")
        st.write(" 1.Derivatives")
        st.write("---->1.1 What is a Derivative?")
        st.write("---->1.2 Rules of Differentiation")
        st.write(" 2.Multivariate Calculus")
        st.write("---->2.1 Total Derivative")
        st.write("---->2.2 Jacobian Matrix")
        st.write("---->2.3 Hessian Matrix")
        st.write(" 3.Neural Networks")
        st.write("---->3.1 Activation Functions")
        st.write(" 4.Taylor Series")
        st.write("---->4.1 Univariate Taylor Series")
        st.write("---->4.2 Multivariate Taylor Series")
        st.write(" 5.Optimization Methods")
        st.write("---->5.1 Gradient Descent")
        st.write("---->5.2 Newton-Raphson Method")
        st.write(" 6.Least Squares - χ² Minimization")

    def derivatives(self):
        st.header('Derivatives')

        st.write("""
        ### What is a Derivative?""")
        st.latex(r""" \textbf{In mathematics, the derivative of a function measures how the function's output value changes as its input changes. It is a fundamental concept in calculus and provides critical insights into the behavior of functions.}""")

        st.write("#### Definition")
        st.latex(r""" \textbf{The derivative of a function \( f \) with respect to \( x \) is defined as:}
        """)
        st.latex(r'''
        f'(x) = \lim_{{\Delta x \to 0}} \frac{{f(x + \Delta x) - f(x)}}{{\Delta x}}
        ''')

        st.header('Rules of Differentiation')

        st.write("""
        ### Overview""")
        st.latex(r"""\textbf{There are several rules in calculus that simplify the process of differentiation:}""")

        st.write("#### Sum Rule")
        st.latex(r'''
        \frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} [f(x)] + \frac{d}{dx} [g(x)]
        ''')

        st.write("""
        #### Power Rule""")
        st.latex(r""" \textbf{For a function of the form \( f(x) = ax^b \), the derivative is given by:}
        """)
        st.latex(r'''
        f'(x) = abx^{b-1}
        ''')

        st.write("""
        #### Product Rule""")
        st.latex(r""" \textbf{For two functions \( f(x) \) and \( g(x) \), the derivative of their product is:}
        """)
        st.latex(r'''
        \frac{d}{dx} [f(x) \cdot g(x)] = f'(x) \cdot g(x) + f(x) \cdot g'(x)
        ''')

        st.write("""
        #### Chain Rule """)
        st.latex(r""" \textbf{The derivative of a composite function is given by:}
        """)
        st.latex(r'''
        \frac{d}{dx} [f(g(x))] = f'(g(x)) \cdot g'(x)
        ''')

    def multivariate_calculus(self):
        st.header('Multivariate Calculus')

        st.write("""
        ### Total Derivative """)
        st.latex(r""" \textbf{The total derivative of a function \( f(x, y, z, \ldots) \) with respect to a parameter \( t \):}
        """)
        st.latex(r'''
        \frac{df}{dt} = \frac{\partial f}{\partial x} \frac{dx}{dt} + \frac{\partial f}{\partial y} \frac{dy}{dt} + \frac{\partial f}{\partial z} \frac{dz}{dt} + \ldots
        ''')

        st.write("""
        ### Jacobian Matrix """)
        st.latex(r""" \textbf{The Jacobian matrix of a vector-valued function \( \mathbf{f}(\mathbf{x}) = (f_1(\mathbf{x}), f_2(\mathbf{x}), \ldots, f_m(\mathbf{x})) \):}""")
        st.latex(r'''
        J_{\mathbf{f}}(\mathbf{x}) = \begin{bmatrix}
        \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
        \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
        \end{bmatrix}
        ''')

        st.write("""
        ### Hessian Matrix """)
        st.latex(r""" \textbf{The Hessian matrix of a scalar-valued function \( f(x_1, x_2, \ldots, x_n) \):}
        """)
        st.latex(r'''
        H_f = \begin{bmatrix}
        \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
        \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
        \vdots & \vdots & \ddots & \vdots \\
        \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
        \end{bmatrix}
        ''')

    def neural_networks(self):
        st.header('Neural Networks')

        st.write("""
        ### Activation Functions""")
        st.latex(r"""  \textbf{In neural networks, activation functions introduce non-linearity to enable complex model learning.}
        """)
        st.latex(r"""\textbf{Sigmoid Function}""")
        st.latex(r'''
        \sigma(x) = \frac{1}{1 + e^{-x}}
        ''')

        st.latex(r"""
         \textbf{Hyperbolic Tangent (Tanh) Function}
        """)
        st.latex(r'''
        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
        ''')

        st.latex(r"""
        \textbf{Rectified Linear Unit (ReLU) Function}
        """)
        st.latex(r'''
        \text{ReLU}(x) = \max(0, x)
        ''')

    def taylor_series(self):
        st.header('Taylor Series')

        st.write("""
        ### Univariate Taylor Series """)
        st.latex(r""" \textbf{The Taylor series expansion of a function \( f(x) \) around a point \( a \):}
        """)
        st.latex(r'''
        f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n
        ''')

        st.write("""
        ### Multivariate Taylor Series""")
        st.latex(r""" \textbf{For functions of multiple variables, the Taylor series expansion around a point \( \mathbf{a} \):}
        """)
        st.latex(r'''
        f(\mathbf{x}) = f(\mathbf{a}) + \nabla f(\mathbf{a}) \cdot (\mathbf{x} - \mathbf{a}) + \frac{1}{2!} (\mathbf{x} - \mathbf{a})^T H_f(\mathbf{a}) (\mathbf{x} - \mathbf{a}) + \ldots
        ''')

    def optimization_methods(self):
        st.header('Optimization Methods')

        st.write("""
        ### Gradient Descent""")
        st.latex(r""" \textbf{Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of the steepest descent.}""")

        st.write("#### Formula")
        st.latex(r'''
        \theta_{i+1} = \theta_i - \gamma \nabla_\theta \chi^2
        ''')

        st.write("""
        ### Newton-Raphson Method """)
        st.latex(r""" \textbf{The Newton-Raphson method is an iterative technique for finding roots of a real-valued function:}""")

        st.write("""#### Formula
        """)
        st.latex(r'''
        x_{i+1} = x_i - \frac{f(x_i)}{f'(x_i)}
        ''')

    def least_squares_minimization(self):
        st.header('Least Squares - χ² Minimization')

        st.write("""
        ### Introduction""")
        st.latex(r""" \textbf{Least squares minimization is a method for fitting a model to data by minimizing the sum of the squares of the differences between observed and predicted values. It's widely used in regression analysis and parameter estimation.}""")

        st.latex(r""" \textbf{For a model \( f(x; \theta) \) with parameters \( \theta \), and data points \( (x_i, y_i) \):}
        """)
        st.latex(r'''
        \chi^2 = \sum_{i=1}^{n} \left( \frac{y_i - f(x_i; \theta)}{\sigma_i} \right)^2
        ''')

        st.latex(r"""
        \textbf{ Where \( \sigma_i \) represents the uncertainties in the measurements.}""")

        st.latex(r""" \textbf{The optimal parameters \( \theta \) are found by minimizing \( \chi^2 \), often using numerical methods like gradient descent or analytical techniques depending on the complexity of the model.}
        """)

    def run(self):
        self.introduction()
        self.includes()
        self.derivatives()
        self.multivariate_calculus()
        self.neural_networks()
        self.taylor_series()
        self.optimization_methods()
        self.least_squares_minimization()
        st.write("### References")
        st.write(r"""
        This section is based on the material from the provided Maths For Machine Learning specialization on Coursera.
                 Thanks to the instructors and the Coursera platform for providing this valuable content.
        """)

def main():
    st.set_page_config(
        page_title="Multivariate Calculus and Applications",
        page_icon=":triangular_ruler:",
        layout="wide"
    )

    app = CalculusApp()
    app.run()

if __name__ == "__main__":
    main()
