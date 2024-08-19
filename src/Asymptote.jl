"""
#  module AsyPlots

A small package to use asymptote (https://asymptote.sourceforge.io) in an
interactive IJulia environment.

PS: This package has nothing to do with the content of my master's thesis.
I included it because it proved very useful in making plots for the actual document. 

## Example
    ```@julia
        asy_code = '''
            import graph;
            import contour;
            usepackage("amssymb");
            size(8cm,8cm);
            real xmin = -2;    real ymin = -2;
            real xmax = 2;     real ymax = 2;

            real f1(real x, real y) {
                return x^2+x*y-y^2;
            };
            real f2(real x, real y) {
                return y^2+x^2-1;
            };
            guide[][] f1var = contour(f1, a=(xmin, ymin), b=(xmax, ymax), new real[] {0});
            guide[][] f2var = contour(f2, a=(xmin, ymin), b=(xmax, ymax), new real[] {0});
            draw(f1var[0], red);
            draw(f2var[0], gray(0.3));
            path[] f1varp = f1var[0];
            path[] f2varp = f2var[0];
            pair[] intersects = intersectionpoints( f1varp,  f2varp);
            for (int i = 0; i < intersects.length; ++i){
                dot(intersects[i], linewidth(4));
            }
            xaxis("x",BottomTop,LeftTicks);
            yaxis("z",LeftRight,RightTicks(trailingzero));
        '''

        plt = AsyPlots.plot(asy_code; mode = "eps"); # create a plot

        display(plt); # show plot in interactive environment

        deliver(plt, "NameOfMyPlot"; extension="pdf"); # save plot in the delivery folder

        set_tmp_folder("/Users/Name/Documents/asymptote") # folder to store intermediate files

        set_exp_folder("/Users/Name/Desktop") # folder to store saved files
    ```
"""
module AsyPlots
import Base.display

tmp_folder = pwd()*"/temp";
exp_folder = pwd();
work_dir = pwd();

mutable struct AsyPkg
    asy_code::String
    file_name::String
    folder::String
end

mutable struct AsyPlot
    asy_code::String
    tmp_folder::String
    file_name::String
    svg_available::Bool
    pdf_available::Bool
    tex_available::Bool
    eps_available::Bool
    ps_available::Bool
    png_available::Bool
    packages::Union{Array{AsyPkg, 1}, Nothing}
end

function set_tmp_folder(folder::String)
    tmp_folder = folder;
end

function set_exp_folder(folder::String)
    tmp_folder = folder;
end

function set_work_folder(folder::String)
    work_dir = folder;
end

function save_as_pkg(asy_code::String, file_name::String; folder::String = "asyscripts")
    cd(work_dir)
    open("$(folder)/$(file_name)_pkg.asy", "w") do io
           write(io, asy_code)
    end
    return AsyPkg(asy_code, file_name, folder)
end

function load_package(file_name::String; folder::String = "asyscripts")
    cd(work_dir)
    asy_code::String = "";
    open("$(folder)/$(file_name)_pkg.asy", "r") do io
           read(io, asy_code)
    end
    return AsyPkg(asy_code, file_name, folder)
end


function set_packages!(plt::AsyPlot, packages::Array{AsyPkg, 1})
    plt.packages = packages;
end

function push_package!(plt::AsyPlot, package::AsyPkg)
    push!(plt.packages, package)
end

function display(plt::AsyPlot)
    cd(work_dir)
    if plt.svg_available
        open("$(plt.tmp_folder)/$(plt.file_name).svg") do f
           display("image/svg+xml", read(f, String))
        end
    elseif plt.png_available
        open("$(plt.tmp_folder)/$(plt.file_name).png") do f
           display("image/png", read(f))
        end
    end
end

function rand_string(len = 6);
    chars = collect("abcdefghijklmnopqrstuvwxyzABSCEFGHIJKLMNOPQRSTUVWXYZ1234567890")
    return String(chars[rand(1:length(chars), len)])
end



function plot(asy_code::String; packages = nothing, file_name = nothing, mode = nothing, viewer = false)
    cd(work_dir)
    if file_name == nothing
        file_name = rand_string()
    end

    # create the tmp_folder
    try; mkdir(tmp_folder); catch e; end

    # create source code file for asy
    open("$(tmp_folder)/$(file_name).txt", "w") do io
            if packages != nothing
                for package ∈ packages
                   write(io, package.asy_code)
                end
            end
            write(io, asy_code)
    end

    svg_available = false
    pdf_available = false
    tex_available = false
    eps_available = false
    ps_available = false
    png_available = false

    plt = AsyPlot(asy_code, tmp_folder, file_name, svg_available, pdf_available, tex_available, eps_available, eps_available, png_available, packages)

    plot!(plt, asy_code; mode = mode, viewer=viewer);

    return plt
end

function plot!(plt::AsyPlot, asy_code::String; mode = nothing, viewer = false)

    #println(pwd());
    cd(work_dir)
    # create source code file for asy
    open("$(plt.tmp_folder)/$(plt.file_name).txt", "w") do io
            if plt.packages != nothing
                for package ∈ plt.packages
                   write(io, package.asy_code)
                end
            end
           write(io, asy_code)
    end

    view_flag = "";
    if viewer
        view_flag = "-V";
    end

    cd("$(plt.tmp_folder)")

    if mode == nothing
        run(`asy $(plt.file_name).txt`)
        run(`epstopdf $(plt.file_name).eps`)
        run(`pdf2svg $(plt.file_name).pdf $(plt.file_name).svg`)
        plt.eps_available=true
        plt.pdf_available=true
        plt.svg_available=true

    elseif mode == "pdf"
        # compile
        run(`asy $(view_flag) -f pdf $(plt.file_name).txt -o $(plt.file_name).pdf`)

        # convert to svg to show in notebook
        run(`pdf2svg $(plt.file_name).pdf $(plt.file_name).svg`)
        plt.svg_available = true
        plt.pdf_available = true
    elseif mode == "svg"
        # compile
        run(`asy $(view_flag) -f svg $(plt.file_name).txt -o $(plt.file_name).svg`)
        plt.svg_available = true
    elseif mode == "png"
        # compile
        run(`asy $(view_flag) -f png $(plt.file_name).txt -o $(plt.file_name).png`)
        plt.png_available = true
    elseif mode == "eps"
        # compile
        run(`asy -V -f eps $(plt.file_name).txt -o $(plt.file_name).eps`)
        # convert to svg to show in notebook
        run(`pdf2svg $(plt.file_name).pdf $(plt.file_name).svg`)
        plt.eps_available = true
        plt.svg_available = true
    end

    cd("../")

    display(plt)

end

function set_delivery_folder(folder::String)
    exp_folder = folder;
end

function deliver(plt::AsyPlot, name::String; extension = "eps", open = true)
    if extension == "eps" && plt.eps_available == false
        run(`asy $(plt.tmp_folder)/$(plt.file_name).txt -o $(plt.tmp_folder)/$(plt.file_name).eps`)
        plt.eps_available = true
    end
    run(`cp $(plt.tmp_folder)/$(plt.file_name).$(extension) $(exp_folder)/$(name).$(extension)`);
    if open
        run(`open $(exp_folder)`);
    end
    return nothing
end


function save_script(asy_code::String, name::String; folder="asyscripts")
    # create source code file for asy
    open("$(folder)/$(name).asy", "w") do io
           write(io, asy_code)
    end
end

function save_script(plt::AsyPlot, name::String; folder="asyscripts")
    # create source code file for asy
    open("$(folder)/$(name).asy", "w") do io
           write(io, plt.asy_code)
    end
end


end
