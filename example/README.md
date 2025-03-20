# Setting up
1. Git clone my repo to whatever place you like: `git clone https://github.com/TzuYaoHuang/InterfaceAdvection.jl.git`
2. checkout to the branch `git checkout demo/VOF`
3. Move to the current folder (`.../example/`)
4. Activate current project (`julia --project=.`)
3. Add my github repo into the project `using Pkg; Pkg.add(url="https://github.com/TzuYaoHuang/InterfaceAdvection.jl", rev="demo/VOF")`
4. Initialize the project: `] Instantiate`
5. Exit Julia and enter it again.
6. Run `using Pluto; Pluto.run(notebook="VOFdemo.jl")`
7. You are free to go!