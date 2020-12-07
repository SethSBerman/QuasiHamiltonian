using LinearAlgebra
using ForwardDiff

workingDim = 3 #Dimension of representation
Eexp(t::Real) = Real[1 t;0 1] #Base matrices
Fexp(t::Real) = Real[1 0;t 1]
Hexp(t::Real) = Real[exp(t) 0;0 exp(-t)]

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
    A = Array{Real}(undef, (2*workingDim,2*workingDim))
        for i = 1:2*workingDim
            for j = 1:2*workingDim
                g = t -> (getValue(i,j)∘(funcMatrix∘K))(t)
                if K == Eexp || K == Fexp
                    c = g(0)
                    a = (g(2) - 2g(1) + c)/2
                    b = g(1) - c - a
                    h = b
                elseif K == Hexp
                    if g(1) != 0
                        l = log(g(1))
                    else
                        l = 0
                    end
                    h = l
                end
                A[i,j]=h
            end
        end
    return A*[x1; x2; x3; y1; y2; y3]
end

function infAction(a::Real,b::Real,c::Real) #Gives the infinitesimal action for a generic matrix
    vec(x1::Real, x2::Real, x3::Real, y1::Real, y2::Real, y3::Real) = b*pointTangent(x1, x2, x3, y1, y2, y3, Eexp) + c*pointTangent(x1, x2, x3, y1, y2, y3, Fexp) + a*pointTangent(x1, x2, x3, y1, y2, y3, Hexp)
    return vec
end

function preMap(a::Real,b::Real,c::Real,x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real) #Gives the value of the symplectic form evaluated on the infinitesimal action and a tangent vector
    vecField = (infAction(a,b,c))(x1,x2,x3,y1,y2,y3)
    wedge = Real[(-1)*vecField[4]   (-1)*vecField[5]    (-1)*vecField[6]    vecField[1] vecField[2] vecField[3]]
    return wedge
end

function momMap(a::Real,b::Real,c::Real) #Finds the function with differential equal to preMap
    grad(x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real) = preMap(a,b,c,x1,x2,x3,y1,y2,y3)
    map(x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real) = (grad(x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real))[1]*x1 + (grad(x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real))[2]*x2 + (grad(x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real))[3]*x3
    return map
end

function momentMapTilda(x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real) #Turns the momemnt map into a function from the cotangent bundle into the dual of the lie algebra
        return (((momMap(0,1,0))(x1,x2,x3,y1,y2,y3))*Real[0 0   1] + ((momMap(0,0,1))(x1,x2,x3,y1,y2,y3))*Real[0    1  0] + (((momMap(1,0,0))(x1,x2,x3,y1,y2,y3)))*Real[(1/2) 0   0])
end

function adEigen(A::Array{<:Real}) #Finds the bad locus in the dual of the lie algebra
    a = A[1]
    b = A[2]
    c = A[3]
    return (a^2+b*c)
end

locus = adEigen ∘ momentMapTilda #Gives the homogenous polynomial which determines the bad locus in the cotangent bundle

test(x1::Real,x2::Real,x3::Real,y1::Real,y2::Real,y3::Real) = (x1*y1)^2 + -2*(x1*y1*y3*x3) + (y3*x3)^2 + ((x2)^2)*y1*y3 + 2*y1*y2*x1*x2 + 2*y2*y3*x2*x3 + 4*((y2)^2)*x1*x3