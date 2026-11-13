-- add_requires pybind11 offline

set_toolchains("gcc")

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match the installed InfiniCore")
    set_values("0", "1")
option_end()

local cxx11_abi = os.getenv("INFINILM_CXX11_ABI") or get_config("cxx11-abi")
if cxx11_abi and cxx11_abi ~= "" then
    if cxx11_abi ~= "0" and cxx11_abi ~= "1" then
        raise("INFINILM_CXX11_ABI must be 0 or 1")
    end
    add_defines("_GLIBCXX_USE_CXX11_ABI=" .. cxx11_abi)
end

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")

-- python / pybind11 头文件：优先环境变量，其次仓库内相对路径（原来写死集群 anaconda 路径）。
-- 同 InfiniCore：不用 pcall、不起子进程——xmake 3.0.7 的脚本沙箱不暴露 pcall。
local function _h3_py_include()
    local e = os.getenv("PYTHON_INCLUDE")
    if e and e ~= "" then
        if os.isdir(e) then return e end
        print("[h3] PYTHON_INCLUDE 指向的目录不存在：" .. e .. "，改为自动探测")
    end
    local prefixes = {}
    for _, k in ipairs({"CONDA_PREFIX", "VIRTUAL_ENV", "PYTHON_ROOT", "PYTHON_HOME"}) do
        local v = os.getenv(k)
        if v and v ~= "" then table.insert(prefixes, v) end
    end
    local home = os.getenv("HOME")
    if home and home ~= "" then table.insert(prefixes, home .. "/.local") end
    for _, p in ipairs({"/usr", "/usr/local", "/opt/conda", "/opt/python"}) do
        table.insert(prefixes, p)
    end
    -- 前缀按优先级排列：先 venv/conda，再系统目录；同一前缀内取版本号最高的
    for _, pre in ipairs(prefixes) do
        if os.isdir(pre) then
            local best, best_key = nil, -1
            for _, d in ipairs(os.dirs(pre .. "/include/python3*") or {}) do
                local maj, mnr = path.filename(d):match("^python(%d+)%.(%d+)$")
                local key = maj and (tonumber(maj) * 1000 + tonumber(mnr)) or 0
                if os.isfile(d .. "/Python.h") and key > best_key then
                    best, best_key = d, key
                end
            end
            if best then return best end
        end
    end
    return nil
end
local _h3_pyinc = _h3_py_include()
if not _h3_pyinc then
    print("[h3] 未探测到 Python 开发头（Python.h）；如需构建 _infinilm，请 export PYTHON_INCLUDE=<含 Python.h 的目录>")
end

local _h3_pbinc = os.getenv("PYBIND11_INCLUDE")
if not _h3_pbinc or _h3_pbinc == "" then
    _h3_pbinc = path.directory(path.directory(os.projectdir())) .. "/third_party/pybind11/include"
end

target("_infinilm")
    add_includedirs(_h3_pbinc, { public = false })
    if _h3_pyinc then add_includedirs(_h3_pyinc, { public = false }) end
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add C++ sources
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
target_end()
