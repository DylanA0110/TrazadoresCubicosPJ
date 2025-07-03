from flask import Flask, render_template, request, jsonify
import numpy as np
from sympy import symbols, diff, lambdify, sympify
import json

app = Flask(__name__, template_folder='templates')
def calcular_trazadores(puntos, condiciones, f_prime_x0=None, f_prime_xn=None):
    """
    Calcula los coeficientes de los trazadores cúbicos
    
    Args:
        puntos: Lista de tuplas (x, y) con los puntos a interpolar
        condiciones: 'natural' o 'sujeta'
        f_prime_x0: Valor de f'(x0) para condiciones sujetas
        f_prime_xn: Valor de f'(xn) para condiciones sujetas
    
    Returns:
        Diccionario con splines, pasos intermedios y resultados
    """
    n = len(puntos) - 1
    x = [p[0] for p in puntos]
    y = [p[1] for p in puntos]
    
    steps = []
    steps.append({'title': 'Datos ingresados', 'content': f'Puntos: {list(zip(x, y))}', 'type': 'info'})
    
    # 1. Calcular h y b
    h = [x[i+1] - x[i] for i in range(n)]
    b = [(y[i+1] - y[i])/h[i] for i in range(n)]
    steps.append({
        'title': 'Diferencias calculadas',
        'content': {'h': [round(val, 4) for val in h], 'b': [round(val, 4) for val in b]},
        'type': 'calculation'
    })

    # 2. Construir sistema matricial
    A = np.zeros((n+1, n+1))
    v = np.zeros(n+1)
    
    if condiciones == 'natural':
        A[0,0] = A[n,n] = 1
        steps.append({'title': 'Condiciones naturales', 'content': "S''(x0) = S''(xn) = 0", 'type': 'condition'})
    else:
        A[0,0], A[0,1] = 2*h[0], h[0]
        v[0] = 3*(b[0] - f_prime_x0)
        A[n,n-1], A[n,n] = h[-1], 2*h[-1]
        v[n] = 3*(f_prime_xn - b[-1])
        steps.append({
            'title': 'Condiciones sujetas',
            'content': f"S'(x0)={f_prime_x0:.4f}, S'(xn)={f_prime_xn:.4f}",
            'type': 'condition'
        })

    for i in range(1, n):
        A[i,i-1:i+2] = h[i-1], 2*(h[i-1]+h[i]), h[i]
        v[i] = 3*(b[i] - b[i-1])
    
    steps.append({
        'title': 'Sistema matricial',
        'content': {'A': [[round(val, 4) for val in row] for row in A.tolist()], 
                   'v': [round(val, 4) for val in v.tolist()]},
        'type': 'matrix'
    })

    # 3. Resolver sistema
    try:
        c = np.linalg.solve(A, v)
    except:
        A_inv = np.linalg.pinv(A)
        c = A_inv @ v
        steps.append({
            'title': 'Advertencia', 
            'content': 'Sistema singular - usando pseudoinversa',
            'type': 'warning'
        })
        steps.append({
            'title': 'Matriz inversa aproximada',
            'content': {'A⁻¹': [[round(val, 6) for val in row] for row in A_inv.tolist()]},
            'type': 'matrix'
        })
    
    steps.append({
        'title': 'Vector solución c',
        'content': {'c': [round(val, 6) for val in c.tolist()]},
        'type': 'calculation'
    })

    # 4. Calcular coeficientes b y d
    b_coeff = []
    d = []
    for i in range(n):
        b_val = (y[i+1]-y[i])/h[i] - h[i]*(2*c[i]+c[i+1])/3
        d_val = (c[i+1] - c[i]) / (3 * h[i])
        b_coeff.append(b_val)
        d.append(d_val)
    
    # 5. Construir splines CORREGIDO (esta es la parte que estaba mal)
    splines = []
    for i in range(n):
        # Aquí estaba el error original - ahora usamos x[i] y x[i+1] correctamente
        x0 = x[i]
        x1 = x[i+1]
        
        spline = {
            'a': y[i],
            'b': b_coeff[i],
            'c': c[i],
            'd': d[i],
            'x0': x0,
            'x1': x1,
            # Expresión corregida usando x0 real
            'expresion': f"{y[i]:.6f} + {b_coeff[i]:.6f}(x - {x0:.4f}) + {c[i]:.6f}(x - {x0:.4f})² + {d[i]:.6f}(x - {x0:.4f})³",
            # Versión alternativa más legible
            'expresion_formateada': f"S_{i}(x) = {y[i]:.6f} + {b_coeff[i]:.6f}(x-{x0:.4f}) + {c[i]:.6f}(x-{x0:.4f})² + {d[i]:.6f}(x-{x0:.4f})³",
            'intervalo': [x0, x1]
        }
        splines.append(spline)
    
    return {
        'success': True,
        'splines': splines,
        'steps': steps,
        'derivadas': {
            'f_prime_x0': f_prime_x0,
            'f_prime_xn': f_prime_xn
        } if condiciones == 'sujeta' else None,
        'solution_vector': [round(val, 6) for val in c.tolist()],
        'puntos_x': x  # Añadido para referencia en el frontend
    }

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/calcular', methods=['POST'])
def calcular():
    try:
        data = request.get_json()
        
        # Validación básica
        if not data or 'funcion' not in data or 'puntos_x' not in data:
            return jsonify({'success': False, 'error': 'Datos incompletos'})

        # Validar puntos_x
        if not isinstance(data['puntos_x'], list):
            return jsonify({'success': False, 'error': 'puntos_x debe ser un array'})
            
        puntos_x = data['puntos_x']
        if len(puntos_x) < 2:
            return jsonify({'success': False, 'error': 'Se requieren mínimo 2 puntos'})
            
        for i in range(1, len(puntos_x)):
            if puntos_x[i] <= puntos_x[i-1]:
                return jsonify({'success': False, 'error': 'Puntos x deben ser crecientes'})

        # Procesar función
        x_sym = symbols('x')
        try:
            func_str = data['funcion'].replace('^', '**')
            f_expr = sympify(func_str)
            f = lambdify(x_sym, f_expr, modules='numpy')
        except Exception as e:
            return jsonify({'success': False, 'error': f'Función inválida: {str(e)}'})

        # Calcular puntos (x, y)
        puntos = [(x, float(f(x))) for x in puntos_x]

        # Manejar condiciones de frontera
        condiciones = data.get('condiciones', 'natural')
        derivadas = {}
        
        if condiciones == 'sujeta':
            try:
                f_prime = diff(f_expr, x_sym)
                f_prime_func = lambdify(x_sym, f_prime, modules='numpy')
                derivadas['f_prime_x0'] = float(f_prime_func(puntos_x[0]))
                derivadas['f_prime_xn'] = float(f_prime_func(puntos_x[-1]))
            except Exception as e:
                return jsonify({'success': False, 'error': f'Error calculando derivadas: {str(e)}'})

        # Calcular trazadores
        try:
            resultados = calcular_trazadores(
                puntos,
                condiciones,
                derivadas.get('f_prime_x0'),
                derivadas.get('f_prime_xn')
            )
            resultados['derivadas'] = derivadas if condiciones == 'sujeta' else None
            return jsonify(resultados)
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error en cálculo: {str(e)}'})

    except Exception as e:
        return jsonify({'success': False, 'error': f'Error inesperado: {str(e)}'})
    
if __name__ == '__main__':
    app.run(debug=True)