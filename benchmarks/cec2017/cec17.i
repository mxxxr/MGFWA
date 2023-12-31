%module cec17

%{
#include "cec17.h"
%}

%typemap(in) (double *x, int nx, int mx, double *f) %{
        if (PyList_Check($input)) {
                $3 = PyList_Size($input);
                $2 = PyList_Size(PyList_GetItem($input, 0)); 
                $1 = (double *) malloc($2 * $3 * sizeof(double));
                $4 = (double *) malloc($3 * sizeof(double));
                int i,j; 
                for (i = 0; i < $3; i++){
                        for (j = 0; j < $2; j++){
                                PyObject *o = PyList_GetItem(PyList_GetItem($input, i), j);
                                double tmp = PyFloat_AsDouble(o);
                                if (PyErr_Occurred())
                                        SWIG_fail;
                                $1[i*$2+j] = PyFloat_AsDouble(o);
                        }
                } 
        } else {
                PyErr_SetString(PyExc_TypeError, "not a list");
                return NULL;
        }
%}

%typemap(freearg) (double *x, int nx, int mx, double *f) %{
        free($1);
        free($4);
%}

%typemap(argout) (double *x, int nx, int mx, double *f) (PyObject* tmp) %{
        tmp = PyList_New($3);
        int i;
        for (i = 0; i < $3; i++){
                PyList_SET_ITEM(tmp, i, PyFloat_FromDouble($4[i]));
        }
        $result = SWIG_Python_AppendOutput($result, tmp);
%}

void eval(double *x, int nx, int mx, double *f, int func_num);
