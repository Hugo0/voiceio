; VoiceIO Inno Setup Installer Script
; Wraps PyInstaller one-dir output into a proper Windows installer.
;
; Prerequisites:
;   1. Run: pyinstaller voiceio.spec
;   2. Verify dist\voiceio\voiceio.exe exists
;   3. Run: iscc installer.iss
;      or:  iscc /DAppVersion=0.2.4 installer.iss

; ---------------------------------------------------------------------------
; Preprocessor defines
; ---------------------------------------------------------------------------
#define AppName      "VoiceIO"
#define AppExeName   "voiceio.exe"
#define AppPublisher "Hugo Montenegro"
#define AppURL       "https://github.com/Hugo0/voiceio"

; Version: override from command line (iscc /DAppVersion=0.2.4 installer.iss)
; or defaults to "dev" for local builds.
#ifndef AppVersion
  #define AppVersion "dev"
#endif

; ---------------------------------------------------------------------------
; [Setup] — core installer metadata and behaviour
; ---------------------------------------------------------------------------
[Setup]
; Unique GUID for this application (generated once, never change it).
AppId={{E7B3F2A1-5C4D-4E8F-9A6B-1D2E3F4A5B6C}

; Display names shown in the installer wizard and Add/Remove Programs
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases

; Install to "C:\Program Files\VoiceIO" by default
DefaultDirName={autopf}\{#AppName}

; Start Menu folder name
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes

; Let the user choose admin vs current-user install
PrivilegesRequiredOverridesAllowed=dialog

; License file shown during install
LicenseFile=LICENSE

; Installer output
OutputDir=dist
OutputBaseFilename=VoiceIO-{#AppVersion}-windows-setup

; Compression
Compression=lzma2
SolidCompression=yes

; Modern flat UI style
WizardStyle=modern

; Tell Windows that this installer may modify PATH
ChangesEnvironment=yes

; Uninstall icon in Add/Remove Programs
UninstallDisplayIcon={app}\{#AppExeName}

; 64-bit install on 64-bit Windows
ArchitecturesInstallIn64BitMode=x64compatible

; ---------------------------------------------------------------------------
; [Languages] — installer UI language
; ---------------------------------------------------------------------------
[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

; ---------------------------------------------------------------------------
; [Tasks] — optional checkboxes shown to the user
; ---------------------------------------------------------------------------
[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath";   Description: "Add VoiceIO to PATH (lets you run 'voiceio' from any terminal)"; GroupDescription: "System integration:"

; ---------------------------------------------------------------------------
; [Files] — what gets copied into the install directory
; ---------------------------------------------------------------------------
; Recursively copy the entire PyInstaller one-dir output.
; "ignoreversion" means always overwrite; "recursesubdirs" handles the
; _internal folder and all bundled DLLs/packages.
[Files]
Source: "dist\voiceio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; ---------------------------------------------------------------------------
; [Icons] — Start Menu and desktop shortcuts
; ---------------------------------------------------------------------------
[Icons]
; Start Menu shortcut (always created)
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"

; Desktop shortcut (only if the user ticked the checkbox)
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

; ---------------------------------------------------------------------------
; [Registry] — PATH modification (per-user or system-wide, matching install scope)
; ---------------------------------------------------------------------------
; When the user selects "Add to PATH", append {app} to the PATH variable.
; Using {code:GetPathRoot} to pick HKCU or HKLM based on install mode.
[Registry]
Root: "HKCU"; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: not IsAdminInstallMode
Root: "HKLM"; Subkey: "SYSTEM\CurrentControlSet\Control\Session Manager\Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}"; Tasks: addtopath; Check: IsAdminInstallMode

; ---------------------------------------------------------------------------
; [Run] — post-install action
; ---------------------------------------------------------------------------
[Run]
Filename: "{app}\{#AppExeName}"; Parameters: "--version"; Description: "Verify installation"; Flags: runhidden nowait postinstall skipifsilent

; ---------------------------------------------------------------------------
; [Code] — Pascal Script for install/uninstall cleanup
; ---------------------------------------------------------------------------
[Code]

// Remove {app} from PATH on uninstall
procedure RemoveFromPath();
var
  Path: String;
  AppDir: String;
  P: Integer;
  RootKey: Integer;
  SubKey: String;
begin
  AppDir := ExpandConstant('{app}');

  if IsAdminInstallMode then
  begin
    RootKey := HKEY_LOCAL_MACHINE;
    SubKey := 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';
  end
  else
  begin
    RootKey := HKEY_CURRENT_USER;
    SubKey := 'Environment';
  end;

  if not RegQueryStringValue(RootKey, SubKey, 'Path', Path) then
    exit;

  // Remove ";{app}" or "{app};" from the PATH string
  P := Pos(';' + AppDir, Path);
  if P > 0 then
  begin
    Delete(Path, P, Length(AppDir) + 1);
    RegWriteExpandStringValue(RootKey, SubKey, 'Path', Path);
  end
  else
  begin
    P := Pos(AppDir + ';', Path);
    if P > 0 then
    begin
      Delete(Path, P, Length(AppDir) + 1);
      RegWriteExpandStringValue(RootKey, SubKey, 'Path', Path);
    end
    else
    begin
      P := Pos(AppDir, Path);
      if P > 0 then
      begin
        Delete(Path, P, Length(AppDir));
        RegWriteExpandStringValue(RootKey, SubKey, 'Path', Path);
      end;
    end;
  end;
end;

// Clean up old files before installing a new version (handles upgrades cleanly)
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssInstall then
  begin
    // Remove the old _internal directory to avoid stale DLLs after upgrade
    if DirExists(ExpandConstant('{app}\_internal')) then
      DelTree(ExpandConstant('{app}\_internal'), True, True, True);
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
  begin
    RemoveFromPath();
  end;
end;
