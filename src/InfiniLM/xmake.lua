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

-- python / pybind11 头文件：优先环境变量，其次仓库内相对路径（原来写死集群 anaconda 路径）
local function _h3_py_include()
    local e = os.getenv("PYTHON_INCLUDE")
    if e and e ~= "" then return e end
    local ok, out = pcall(os.iorunv, "python3", {"-c",
        "import sysconfig;print(sysconfig.get_paths()['include'])"})
    if ok and out then
        out = out:gsub("%s+$", "")
        if out ~= "" and os.isdir(out) then return out end
    end
    return nil
end
local _h3_pyinc = _h3_py_include()

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
