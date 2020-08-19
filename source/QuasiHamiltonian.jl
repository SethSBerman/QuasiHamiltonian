using LinearAlgebra
using ForwardDiff

workingDim = 3 #Dimension of representation
Eexp(t::Real) = [1 t;0 1] #Base matrices
Fexp(t::Real) = [1 0;t 1]
Hexp(t::Real) = [exp(t) 0;0 exp(-t)]

function rep(A::Array{<:Real}) #Gives the representation
    a=A[1,1]
    b=A[1,2]
    c=A[2,1]
    d=A[2,2]
    return Real[a^2 a*b b^2; 2*a*c (c*b)+(a*d) 2*b*d; c^2 c*d d^2]
end

function funcMatrix(A::Array{<:Real}) #Provides the action on the cotangent bundle
    return Real[rep(A) zeros(Real, 3, 3);zeros(Real, 3, 3) transpose(rep(inv(A)))]
end

function getValue(j, k) #Used for composition
    f(A) = A[j,k]
    return f
end

function pointTangent(x1::Real, x2::Real, x3::Real, y1::Real, y2::Real, y3::Real, K) #Gives the infinitesimal action for a given base matrix
    A = Array{Int64}(undef, (2*workingDim,2*workingDim))
        for i = 1:2*workingDim
            for j = 1:2*workingDim
                g = t -> (getValue(i,j)∘(funcMatrix∘K))(t)
                h = ForwardDiff.derivative(g, 0)
                A[i,j]=h
            end
        end
    return A*[x1; x2; x3; y1; y2; y3]
end

function infAction(a::Real,b::Real,c::Real) #Gives the infinitesimal action for a generic matrix
    vec(x1::Real, x2::Real, x3::Real, y1::Real, y2::Real, y3::Real) = b*pointTangent(x1, x2, x3, y1, y2, y3, Eexp) + c*pointTangent(x1, x2, x3, y1, y2, y3, Fexp) + a*pointTangent(x1, x2, x3, y1, y2, y3, Hexp)
    return vec
end





