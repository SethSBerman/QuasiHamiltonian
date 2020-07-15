using LinearAlgebra

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

print(funcMatrix(Complex[1 1;0 1]))
