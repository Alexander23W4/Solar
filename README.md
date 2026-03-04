# Solar
极地光伏组件及其阵列设计项目

### 项目建模源码
./src
* AssemblyModule 为多板整合成组件的建模实现
* AngleModule  用于计算组件当中不同倾角和方位角的光伏板（即光伏板法相角）与太阳直射角的夹角建模
* IrradianceModule 用于计算直射irradiance 散射irradiance 反射irradiance 建模实现
* Array 用于组件阵列的建模
* Timeline 用于计算阵列在某一南极地点某一段时间理想状态下的大致总irradiance， 此建模用于对时间积分

### 资料
./research_materials
包含参考论文  项目报告与展示文件  项目设计方案与规划

### 测试
./test
测试文件在此处， 用于测试不同方案的irradiance结果

