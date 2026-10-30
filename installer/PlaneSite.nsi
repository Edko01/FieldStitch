; PlaneSite Windows installer (NSIS 3, Modern UI 2)
; Build:  makensis -DVERSION=1.2.0 -DSTAGE=<folder with python\ app\ PlaneSite.ico README.txt> -DOUTFILE=<setup.exe> PlaneSite.nsi
;
; Per-user install (no admin prompt) so in-app updates can replace files without elevation.

Target amd64-unicode
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!define APPNAME "PlaneSite"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\PlaneSite"

Name "${APPNAME}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\PlaneSite"
InstallDirRegKey HKCU "Software\PlaneSite" "InstallDir"
RequestExecutionLevel user
BrandingText "PlaneSite ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "PlaneSite"
VIAddVersionKey "FileDescription" "PlaneSite Setup"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "PlaneSite"

!include "MUI2.nsh"
!include "FileFunc.nsh"

!define MUI_ICON "${STAGE}\PlaneSite.ico"
!define MUI_UNICON "${STAGE}\PlaneSite.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEFINISHPAGE_BITMAP "${STAGE}\welcome.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${STAGE}\welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "${STAGE}\header.bmp"
!define MUI_HEADERIMAGE_UNBITMAP "${STAGE}\header.bmp"
!define MUI_WELCOMEPAGE_TITLE "Install PlaneSite ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "PlaneSite builds georeferenced orthomosaics from drone mapping photos and splits them into field GeoTIFFs. Processing runs entirely on this PC; no internet connection is needed.$\r$\n$\r$\nIf PlaneSite is open, Setup will close it. Your projects are not affected.$\r$\n$\r$\nPlaneSite was called FieldStitch before version 1.6. If FieldStitch is installed, Setup replaces it and keeps its settings.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Open PlaneSite"
!define MUI_FINISHPAGE_RUN_FUNCTION LaunchApp
!define MUI_FINISHPAGE_SHOWREADME "$INSTDIR\README.txt"
!define MUI_FINISHPAGE_SHOWREADME_TEXT "Show the quick guide"
!define MUI_FINISHPAGE_SHOWREADME_NOTCHECKED

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function LaunchApp
  SetOutPath "$INSTDIR"
  Exec '"$INSTDIR\python\PlaneSiteW.exe" "$INSTDIR\app\main.py"'
FunctionEnd

; PlaneSite was called FieldStitch before 1.6: close it and uninstall it quietly (its settings stay in
; %APPDATA%\FieldStitch and PlaneSite copies them on first start; projects are not touched)
!define OLDKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\FieldStitch"
Function RemoveFieldStitch
  ReadRegStr $0 HKCU "${OLDKEY}" "InstallLocation"
  StrCmp $0 "" 0 +2
    StrCpy $0 "$LOCALAPPDATA\Programs\FieldStitch"
  IfFileExists "$0\app\main.py" 0 shortcuts
  IfFileExists "$0\python\python.exe" 0 shortcuts
  DetailPrint "Closing FieldStitch"
  nsExec::Exec '"$0\python\python.exe" "$0\app\main.py" --quit'
  Pop $1
  Sleep 2000
  IfFileExists "$0\Uninstall FieldStitch.exe" 0 manual
  DetailPrint "Removing FieldStitch (replaced by PlaneSite)"
  ExecWait '"$0\Uninstall FieldStitch.exe" /S _?=$0'
  Delete "$0\Uninstall FieldStitch.exe"
  manual:
  RMDir /r "$0\app"
  RMDir /r "$0\python"
  RMDir /r "$0\_update"
  Delete "$0\FieldStitch.ico"
  Delete "$0\README.txt"
  RMDir "$0"
  shortcuts:
  Delete "$SMPROGRAMS\FieldStitch.lnk"
  Delete "$DESKTOP\FieldStitch.lnk"
  DeleteRegKey HKCU "${OLDKEY}"
  DeleteRegKey HKCU "Software\FieldStitch"
  RMDir /r "$LOCALAPPDATA\FieldStitch"   ; old program logs and window data
FunctionEnd

Function CloseRunning
  IfFileExists "$INSTDIR\python\python.exe" 0 done
  IfFileExists "$INSTDIR\app\main.py" 0 done
  nsExec::Exec '"$INSTDIR\python\python.exe" "$INSTDIR\app\main.py" --quit'
  Pop $0
  Sleep 2000
  done:
FunctionEnd

Section "PlaneSite (required)" SecMain
  SectionIn RO
  Call RemoveFieldStitch
  Call CloseRunning
  ; replace program files cleanly (projects live elsewhere)
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\_update"
  SetOutPath "$INSTDIR"
  File /r "${STAGE}\python"
  File /r "${STAGE}\app"
  File "${STAGE}\PlaneSite.ico"
  File "${STAGE}\README.txt"
  WriteUninstaller "$INSTDIR\Uninstall PlaneSite.exe"

  SetOutPath "$INSTDIR"
  CreateShortCut "$SMPROGRAMS\PlaneSite.lnk" "$INSTDIR\python\PlaneSiteW.exe" '"$INSTDIR\app\main.py"' \
    "$INSTDIR\PlaneSite.ico" 0 SW_SHOWNORMAL "" "Offline orthomosaics from drone photos"

  WriteRegStr HKCU "Software\PlaneSite" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "PlaneSite"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "PlaneSite"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\PlaneSite.ico"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Uninstall PlaneSite.exe"'
  WriteRegStr HKCU "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\Uninstall PlaneSite.exe" /S'
  WriteRegStr HKCU "${UNINSTKEY}" "URLInfoAbout" "https://github.com/EdkoLLC/PlaneSite"
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "${UNINSTKEY}" "EstimatedSize" "$0"
SectionEnd

Section "Desktop shortcut" SecDesktop
  SetOutPath "$INSTDIR"
  CreateShortCut "$DESKTOP\PlaneSite.lnk" "$INSTDIR\python\PlaneSiteW.exe" '"$INSTDIR\app\main.py"' \
    "$INSTDIR\PlaneSite.ico" 0 SW_SHOWNORMAL "" "Offline orthomosaics from drone photos"
SectionEnd

LangString DESC_SecMain ${LANG_ENGLISH} "The PlaneSite program and its processing engine."
LangString DESC_SecDesktop ${LANG_ENGLISH} "Put a PlaneSite icon on the desktop."
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} $(DESC_SecMain)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} $(DESC_SecDesktop)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  IfFileExists "$INSTDIR\python\python.exe" 0 +3
    nsExec::Exec '"$INSTDIR\python\python.exe" "$INSTDIR\app\main.py" --quit'
    Sleep 2000
  Delete "$SMPROGRAMS\PlaneSite.lnk"
  Delete "$DESKTOP\PlaneSite.lnk"
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\_update"
  Delete "$INSTDIR\PlaneSite.ico"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\Uninstall PlaneSite.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\PlaneSite"
  ; projects (Documents\PlaneSite Projects) and settings are left in place
SectionEnd
