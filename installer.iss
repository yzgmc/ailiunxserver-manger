; Inno Setup 脚本 —— Linux Server AI Agent 安装程序
; 编译：ISCC.exe installer.iss
; 产物：installer\LinuxServerAIAgent-1.0.0-setup.exe

#define MyAppName "Linux Server AI Agent"
#define MyAppNameZh "Linux 服务器管理 AI 代理"
#define MyAppVersion "1.0.0"
#define MyAppExeName "LinuxServerAIAgent.exe"

[Setup]
AppId={{6CC9F287-A539-4EC6-ACDE-4149F2E64346}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppName}
DefaultDirName={autopf}\LinuxServerAIAgent
DefaultGroupName={#MyAppName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}
; 按用户安装：无需管理员权限，不弹 UAC
PrivilegesRequired=lowest
OutputDir=installer
OutputBaseFilename=LinuxServerAIAgent-{#MyAppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

[Languages]
Name: "chs"; MessagesFile: "ChineseSimplified.isl"
Name: "enu"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\LinuxServerAIAgent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppNameZh}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\卸载 {#MyAppNameZh}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppNameZh}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; 清理运行时在安装目录内可能产生的残留
Type: filesandordirs; Name: "{app}\.linux-server-agent"
