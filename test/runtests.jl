using Test

using Distributions

@time using YPPL_Parser

using YPPL_Parser: @build_likeli_inline, @build_likeli, build_likeli_quote_inline, build_likeli_quote
using YPPL_Parser.Examples.eight_schools_non_centered
using YPPL_Parser.Examples.ref_eight_schools_non_centered


@testset "YPPL_Parser" begin

    likeli = @build_likeli(
        eight_schools_non_centered.ex, 
        eight_schools_non_centered.p, 
        eight_schools_non_centered.schools_dat
    )


    likeli_inline = @build_likeli_inline(
        eight_schools_non_centered.ex, 
        eight_schools_non_centered.p, 
        eight_schools_non_centered.schools_dat
    )

    likeli_built = eight_schools_non_centered.likeli

    likeli_ori = ref_eight_schools_non_centered.likeli

    res_arr = [l(ones(10)) for l in [likeli, likeli_inline, likeli_built, likeli_ori]]
    println(res_arr)
    for i in 2:length(res_arr)
        @test res_arr[1] == res_arr[i]
    end
end