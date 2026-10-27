; FieldStitch Windows installer (NSIS 3, Modern UI 2)
; Build:  makensis -DVERSION=1.2.0 -DSTAGE=<folder with python\ app\ FieldStitch.ico README.txt> -DOUTFILE=<setup.exe> FieldStitch.nsi
;
; Per-user install (no admin prompt) so in-app updates can replace files without elevation.

Target amd64-unicode
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!define APPNAME "FieldStitch"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\FieldStitch"

Name "${APPNAME}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\FieldStitch"
InstallDirRegKey HKCU "Software\FieldStitch" "InstallDir"
RequestExecutionLevel user
BrandingText "FieldStitch ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "FieldStitch"
VIAddVersionKey "FileDescription" "FieldStitch Setup"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "FieldStitch"

!include "MUI2.nsh"
!include "FileFunc.nsh"

!define MUI_ICON "${STAGE}\FieldStitch.ico"
!define MUI_UNICON "${STAGE}\FieldStitch.ico"
!define MUI_ABORTWARNING
!define MUI_WELCOMEFINISHPAGE_BITMAP "${STAGE}\welcome.bmp"
!define MUI_UNWELCOMEFINISHPAGE_BITMAP "${STAGE}\welcome.bmp"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_RIGHT
!define MUI_HEADERIMAGE_BITMAP "${STAGE}\header.bmp"
!define MUI_HEADERIMAGE_UNBITMAP "${STAGE}\header.bmp"
!define MUI_WELCOMEPAGE_TITLE "Install FieldStitch ${VERSION}"
!define MUI_WELCOMEPAGE_TEXT "FieldStitch builds georeferenced orthomosaics from drone mapping photos and splits them into field GeoTIFFs. Processing runs entirely on this PC; no internet connection is needed.$\r$\n$\r$\nIf FieldStitch is open, Setup will close it. Your projects are not affected.$\r$\n$\r$\nClick Next to continue."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Open FieldStitch"
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
  Exec '"$INSTDIR\python\pythonw.exe" "$INSTDIR\app\main.py"'
FunctionEnd

Function CloseRunning
  IfFileExists "$INSTDIR\python\python.exe" 0 done
  IfFileExists "$INSTDIR\app\main.py" 0 done
  nsExec::Exec '"$INSTDIR\python\python.exe" "$INSTDIR\app\main.py" --quit'
  Pop $0
  Sleep 2000
  done:
FunctionEnd

Section "FieldStitch (required)" SecMain
  SectionIn RO
  Call CloseRunning
  ; replace program files cleanly (projects live elsewhere)
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\_update"
  SetOutPath "$INSTDIR"
  File /r "${STAGE}\python"
  File /r "${STAGE}\app"
  File "${STAGE}\FieldStitch.ico"
  File "${STAGE}\README.txt"
  WriteUninstaller "$INSTDIR\Uninstall FieldStitch.exe"

  SetOutPath "$INSTDIR"
  CreateShortCut "$SMPROGRAMS\FieldStitch.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\main.py"' \
    "$INSTDIR\FieldStitch.ico" 0 SW_SHOWNORMAL "" "Offline orthomosaics from drone photos"

  WriteRegStr HKCU "Software\FieldStitch" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "FieldStitch"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "FieldStitch"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\FieldStitch.ico"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\Uninstall FieldStitch.exe"'
  WriteRegStr HKCU "${UNINSTKEY}" "QuietUninstallString" '"$INSTDIR\Uninstall FieldStitch.exe" /S'
  WriteRegStr HKCU "${UNINSTKEY}" "URLInfoAbout" "https://github.com/Edko01/FieldStitch"
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD HKCU "${UNINSTKEY}" "EstimatedSize" "$0"
SectionEnd

Section "Desktop shortcut" SecDesktop
  SetOutPath "$INSTDIR"
  CreateShortCut "$DESKTOP\FieldStitch.lnk" "$INSTDIR\python\pythonw.exe" '"$INSTDIR\app\main.py"' \
    "$INSTDIR\FieldStitch.ico" 0 SW_SHOWNORMAL "" "Offline orthomosaics from drone photos"
SectionEnd

LangString DESC_SecMain ${LANG_ENGLISH} "The FieldStitch program and its processing engine."
LangString DESC_SecDesktop ${LANG_ENGLISH} "Put a FieldStitch icon on the desktop."
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecMain} $(DESC_SecMain)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} $(DESC_SecDesktop)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

Section "Uninstall"
  IfFileExists "$INSTDIR\python\python.exe" 0 +3
    nsExec::Exec '"$INSTDIR\python\python.exe" "$INSTDIR\app\main.py" --quit'
    Sleep 2000
  Delete "$SMPROGRAMS\FieldStitch.lnk"
  Delete "$DESKTOP\FieldStitch.lnk"
  RMDir /r "$INSTDIR\app"
  RMDir /r "$INSTDIR\python"
  RMDir /r "$INSTDIR\_update"
  Delete "$INSTDIR\FieldStitch.ico"
  Delete "$INSTDIR\README.txt"
  Delete "$INSTDIR\Uninstall FieldStitch.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\FieldStitch"
  ; projects (Documents\FieldStitch Projects) and settings are left in place
SectionEnd
