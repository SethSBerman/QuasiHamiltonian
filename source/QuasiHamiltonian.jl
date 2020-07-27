using LinearAlgebra

E = Complex[0 1;0 0]
F = Complex[0 0;1 0]
H = Complex[1 0;0 -1]

function rep(A)
    a=A[1,1]
    b=A[1,2]
    c=A[2,1]
    d=A[2,2]
    return Complex[a^2 a*b b^2; 2*a*c (c*b)+(a*d) 2*b*d; c^2 c*d d^2]
end

function funcMatrix(A)
    return Complex[rep(A) zeros(Complex, 3, 3);zeros(Complex, 3, 3) transpose(rep(inv(A)))]
end

function scalarEx(B)
    exponentFunc = function(t::Complex) return exp(t*B) end
    return exponentFunc
end

function pointTangent(x1, x2, x3, y1, y2, y3, K)
    timesPoint = function(D) return D*Complex[x1; x2; x3; y1; y2; y3] end
    return #= derivative =# (timesPoint∘(funcMatrix∘(scalarEx(K))))
end



