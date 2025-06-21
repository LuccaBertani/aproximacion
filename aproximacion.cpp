#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>  // Para log y exp

using namespace std;

// Función para calcular coeficientes de mínimos cuadrados lineales (y = m*x + b)
void calcularRectaMinimosCuadrados(const vector<double>& x, const vector<double>& y, double& m, double& b) {
    int n = x.size();
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;

    for (int i = 0; i < n; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }

    double denom = n * sumX2 - sumX * sumX;
    if (denom == 0) {
        cerr << "Error: división por cero en cálculo de mínimos cuadrados." << endl;
        m = 0;
        b = 0;
        return;
    }
    m = (n * sumXY - sumX * sumY) / denom;
    b = (sumY - m * sumX) / n;
}

// Calcula ECM y R^2 para modelo dado (x, y) y función predicha f(x)
void calcularErrorYCoefR2(const vector<double>& y, const vector<double>& y_pred, double& ecm, double& r2) {
    int n = y.size();
    double sumaErroresCuadrados = 0;
    double sumaTotalesCuadrados = 0;
    double sumaY = 0;

    for (double val : y) sumaY += val;
    double mediaY = sumaY / n;

    for (int i = 0; i < n; ++i) {
        double error = y[i] - y_pred[i];
        sumaErroresCuadrados += error * error;

        double total = y[i] - mediaY;
        sumaTotalesCuadrados += total * total;
    }

    ecm = sumaErroresCuadrados / n;
    r2 = 1 - (sumaErroresCuadrados / sumaTotalesCuadrados);
}

int main(int argc, char* argv[]) {
    if(argc != 2){
        cout << "Ingresar con este formato: ./aproximacion <nombre_archivo_csv>" << endl;
        return -1;
    }
    string nombreArchivo = argv[1];
    ifstream archivo(nombreArchivo);
    if (!archivo.is_open()) {
        cerr << "No se pudo abrir el archivo: " << nombreArchivo << endl;
        return 1;
    }

    vector<double> horas;
    vector<double> calificaciones;
    string linea;

    while (getline(archivo, linea)) {
        stringstream ss(linea);
        string token;
        double hora, calif;

        // Leer horas
        if (!getline(ss, token, ',')) continue;
        try {
            hora = stod(token);
        } catch (...) {
            cerr << "Error al convertir horas: " << token << endl;
            continue;
        }

        // Leer calificación
        if (!getline(ss, token, ',')) continue;
        try {
            calif = stod(token);
        } catch (...) {
            cerr << "Error al convertir calificación: " << token << endl;
            continue;
        }

        horas.push_back(hora);
        calificaciones.push_back(calif);
    }

    archivo.close();

    if (horas.size() == 0) {
        cerr << "No se leyeron datos válidos." << endl;
        return 1;
    }

    // --- Modelo lineal ---
    double m_lin, b_lin;
    calcularRectaMinimosCuadrados(horas, calificaciones, m_lin, b_lin);

    vector<double> y_pred_lineal;
    for (double x : horas) y_pred_lineal.push_back(m_lin * x + b_lin);

    double ecm_lin, r2_lin;
    calcularErrorYCoefR2(calificaciones, y_pred_lineal, ecm_lin, r2_lin);

    cout << "Modelo Lineal: y = " << m_lin << " * x + " << b_lin << endl;
    cout << "ECM Lineal: " << ecm_lin << ", R^2 Lineal: " << r2_lin << endl << endl;

    // --- Modelo exponencial: y = a * exp(b * x)
    // Linealizamos: ln(y) = ln(a) + b*x
    vector<double> ln_y;
    vector<double> x_exp;
    for (int i = 0; i < (int)calificaciones.size(); ++i) {
        if (calificaciones[i] <= 0) {
            cerr << "Datos no válidos para logaritmo (y <= 0) en índice " << i << ", se omite." << endl;
            continue;
        }
        x_exp.push_back(horas[i]);
        ln_y.push_back(log(calificaciones[i]));
    }

    if (ln_y.size() < 2) {
        cerr << "No hay suficientes datos válidos para ajuste exponencial." << endl;
    } else {
        double m_exp, b_exp;
        calcularRectaMinimosCuadrados(x_exp, ln_y, b_exp, m_exp); // Nota: m_exp es intercepto ln(a), b_exp es pendiente b

        double a_exp = exp(m_exp);

        vector<double> y_pred_exp;
        for (double x : horas) {
            y_pred_exp.push_back(a_exp * exp(b_exp * x));
        }

        double ecm_exp, r2_exp;
        calcularErrorYCoefR2(calificaciones, y_pred_exp, ecm_exp, r2_exp);

        cout << "Modelo Exponencial: y = " << a_exp << " * exp(" << b_exp << " * x)" << endl;
        cout << "ECM Exponencial: " << ecm_exp << ", R^2 Exponencial: " << r2_exp << endl << endl;
    }

    // --- Modelo logarítmico: y = a + b * ln(x)
    // Solo x > 0
    vector<double> ln_x;
    vector<double> y_log;
    for (int i = 0; i < (int)horas.size(); ++i) {
        if (horas[i] <= 0) {
            cerr << "Datos no válidos para logaritmo (x <= 0) en índice " << i << ", se omite." << endl;
            continue;
        }
        ln_x.push_back(log(horas[i]));
        y_log.push_back(calificaciones[i]);
    }

    if (ln_x.size() < 2) {
        cerr << "No hay suficientes datos válidos para ajuste logarítmico." << endl;
    } else {
        double m_log, b_log;
       calcularRectaMinimosCuadrados(ln_x, y_log, m_log, b_log); // b_log es intercepto a, m_log es pendiente b

        vector<double> y_pred_log;
        for (double x : horas) {
            if (x <= 0) {
                y_pred_log.push_back(0); // O manejar de otra forma
            } else {
                y_pred_log.push_back(b_log + m_log * log(x));
            }
        }

        double ecm_log, r2_log;
        calcularErrorYCoefR2(calificaciones, y_pred_log, ecm_log, r2_log);

        cout << "Modelo Logarítmico: y = " << b_log << " + " << m_log << " * ln(x)" << endl;
        cout << "ECM Logarítmico: " << ecm_log << ", R^2 Logarítmico: " << r2_log << endl << endl;
    }

    return 0;
}