c  These subroutines implement tabulated rates for weak processes
c  positron decay(emissiion), electron capture, positron capture, electron decay(emission)
c
c  Dean Townsley 2009
c
c  utilizes tables compiled by Ed Brown from the literature
c  passed-in structures are those of Frank Timmes' torch reaction network code


c  There are two callable subroutines
c  weaktab_rates(..) fills the passed structures with the relevant raw
c                     rates (=still need to be multiplied by abundance)
c  weaktab_init(..)  reads table from file and sets up data structures
c                     can be called again to change nuclide set
c                     re-reads table from file to reconstruct memory tables
c
c  Not yet implemented:
c  weaktab_nuloss(..) calculates the energy loss rate given temperature
c                     and a set of abundances

c  The rate tables are stored in a module data area that is not used outside
c  these routines


      module weaktab
      implicit none
       
c      nuclide table indices of proton and neutron
      integer :: pnuci, nnuci

c      for efficiency we will statically assume a FFN grid
c      which notably uses an integer logarithmic grid for electron density
c      This has 13 temperature values, in K
      integer,parameter :: ntemp = 13
      double precision :: tempgrid(ntemp) =
     1    (/ 0.01e9, 0.1e9, 0.2e9, 0.4e9, 0.7e9, 1.0e9,
     2       1.5e9 , 2.0e9, 3.0e9, 5.0e9, 10.0e9, 30.0e9, 100.0e9 /)
c      and 11 values of log10 of e^- density
c      (in units of N_A per cm^3 ~= Ye*mass density in g/cc)
      integer, parameter :: nedens = 11
      double precision :: ledens_min=1.0, ledens_max=11.0

c table storage format in memory
c  separate arrays for e^+ decay (emission), e^- capture, pd-ec neutrino loss rates
c                      e^- decay (emission), e^+ capture, ed-pc neutrino loss rates
c  Entries at each density, temperature are stored by target nuclide in the same
c  order as the nuclide table ( aion(:), zion(:) etc ).
c  Entries at one density and temperature for all nuclides are stored contiguously to improve
c  cache-locality during an evalutation at a given density and temperature.
      logical :: tables_initialized = .false.
      integer :: numnuc
      integer, allocatable :: map_nuci_to_lepi(:),map_nuci_to_delepi(:)
      double precision, dimension(:,:,:), allocatable ::
     1       wtab_pd, wtab_ec, wtab_pdecnu,
     2       wtab_ed, wtab_pc, wtab_edpcnu

      integer, dimension(:), allocatable :: lep_prod, delep_prod

      integer :: tempi_saved

      save

      private nucname2num
      contains

c =========================================================================

c      subroutine to initialize memory tables from file
c      Nuclide indexes are derived to match the indexing of the zion and
c      aion arrays passed in.
c      The in-memory tables have one entry per target, but we don't know
c      how many targets we have rates for until we read in the table and
c      compare all the entries to our list of included nuclides.  So the
c      read is done in two stages.  First the rates are read into sparse
c      arrays that have an entry for each included nuclide.  A map is then
c      constructed from nuclide index to a sequence in which nuclides
c      lacking rates do not appear.  There is a separate sequence for 
c      leptonization and deleptonization rates, since a given target may
c      have one and not the other.  Finally the rate tables are re-formatted
c      into a compact form based on this compact sequence.
c
c      Why? -- we want the rate tables to be in the same order as the
c      nuclide table, since that is the order the iteration will take
c      during rate calculation.  This improves cache locality.  Further,
c      the tables should be dense for reasons of both cache locality and
c      memory footprint.  We could scan the file twice instead of using
c      the sparse arrays.
      subroutine weaktab_init(netnumnuc,zion,aion)

      implicit none

      integer, intent(in) :: netnumnuc
      double precision,intent(in):: zion(:), aion(:)

c      sparse arrays for reading in data tables
      double precision, dimension(:,:,:), allocatable ::
     1       s_wtab_pd, s_wtab_ec, s_wtab_pdecnu,
     2       s_wtab_ed, s_wtab_pc, s_wtab_edpcnu

      integer :: i, j, edensi, tempi, delepi, lepi, weakfile, stat
      integer :: z1, a1, z2, a2, nuci1, nuci2
      integer :: numlep, numdelep

      character(len=5) :: name1, name2
      character(len=1024) :: buf

      double precision :: temp, ledens

c     check if this is an init or a re-init
c     if it is a re-init, clear current tables
      if (tables_initialized) then
       numnuc=0
       deallocate( lep_prod )
       deallocate( delep_prod )
       deallocate( map_nuci_to_lepi )
       deallocate( map_nuci_to_delepi )
       deallocate( wtab_pd )
       deallocate( wtab_ec )
       deallocate( wtab_pdecnu )
       deallocate( wtab_ed )
       deallocate( wtab_pc )
       deallocate( wtab_edpcnu )
      else
       tables_initialized = .true.
      endif

c     allocate sparse tables based on number of nuclides
      numnuc=netnumnuc
      allocate( lep_prod(numnuc) )
      allocate( delep_prod(numnuc) )
      allocate( s_wtab_pd(numnuc,ntemp,nedens) )
      allocate( s_wtab_ec(numnuc,ntemp,nedens) )
      allocate( s_wtab_pdecnu(numnuc,ntemp,nedens) )
      allocate( s_wtab_ed(numnuc,ntemp,nedens) )
      allocate( s_wtab_pc(numnuc,ntemp,nedens) )
      allocate( s_wtab_edpcnu(numnuc,ntemp,nedens) )

c      save neutron and proton indices for later use
       do i=1,numnuc
        if ( int(aion(i)).eq. 1 ) then
         if ( int(zion(i)).eq. 0 ) then
          nnuci=i
         else
          pnuci=i
         endif
        endif
       enddo

c      -----------  Now read table entries

c     will use to accumulate which nuclides we have rates for (as "targets")
c     Different for leptonization (ed pc) and deleptonization (pd ec) directions
c     will also use later to get products when iterating through targets
      lep_prod(:) = -1
      delep_prod(:) = -1

c     open file
      weakfile = 33
      open( unit=weakfile, file='weak_rates', status='old', iostat=stat)
      if (stat.ne.0) then
       print *, 'unable to open weak rates file'
       stop
      endif

c     read first line then iterate through file
c     read nuclide names of leptonization pair
      read(weakfile,"(a)",iostat=stat) buf
      do while ( stat .ge. 0 )

c      find end of first name on line
       do i=1,1024
        if (buf(i:i)==' ') exit
       enddo
c      find beginning of second name on line
       do j=i+1,1024
        if (buf(j:j)/=' ') exit
       enddo
       if (i>100.or.j>100) stop 'unable parse leptonization pair names'
       name1 = buf(1:i-1)
       name2 = buf(j:j+4)
       call nucname2num(name1, z1,a1)
       call nucname2num(name2, z2,a2)

c      look for nuclide indices
       nuci1 = -1
       nuci2 = -1
       do i=1,numnuc
        if ( int(zion(i)).eq.z1 .and. int(aion(i)).eq.a1 ) then
         nuci1 = i
         exit
        endif
       enddo
       do i=1,numnuc
        if ( int(zion(i)).eq.z2 .and. int(aion(i)).eq.a2 ) then
         nuci2 = i
         exit
        endif
       enddo

c      read in rate table
c      Table file format:
c        One table for each nuclide pair The table is gridded in temperature
c        and electron density.
       if ( nuci1 .ne. -1 .and. nuci2 .ne. -1 ) then
        do edensi=1,nedens
         do tempi=1,ntemp
          read(weakfile,*) temp, ledens,
     1                       s_wtab_pd(nuci1,tempi,edensi),
     2                       s_wtab_ec(nuci1,tempi,edensi),
     3                       s_wtab_pdecnu(nuci1,tempi,edensi),
     4                       s_wtab_ed(nuci2,tempi,edensi),
     5                       s_wtab_pc(nuci2,tempi,edensi),
     6                       s_wtab_edpcnu(nuci2,tempi,edensi)
          delep_prod(nuci1) = nuci2
          lep_prod(nuci2) = nuci1
         enddo
        enddo
       else
        do edensi=1,nedens
         do tempi=1,ntemp
          read(weakfile,"(a)") buf
         enddo
        enddo
       endif

c      eat the "@" after each entry
       read(weakfile,"(a)") buf
          
c      read next leptonization pair
       read(weakfile,"(a)",iostat=stat) buf
      enddo
      close(weakfile)

c     -------  Now pack the tables and create index maps

c     count lep and delep targets
c     and create map from nuclide index to index in lep/delep tables
      allocate(map_nuci_to_lepi(numnuc))
      allocate(map_nuci_to_delepi(numnuc))
      numlep = 0
      numdelep = 0
      do i=1,numnuc
       if ( lep_prod(i) > 0 ) then
        numlep = numlep + 1
        map_nuci_to_lepi(i) = numlep
       else
        map_nuci_to_lepi(i) = -1
       endif
       if ( delep_prod(i) > 0 ) then
        numdelep = numdelep + 1
        map_nuci_to_delepi(i) = numdelep
       else
        map_nuci_to_delepi(i) = -1
       endif
      enddo
c     allocate compact tables
      allocate( wtab_pd(numdelep,ntemp,nedens) )
      allocate( wtab_ec(numdelep,ntemp,nedens) )
      allocate( wtab_pdecnu(numdelep,ntemp,nedens) )
      allocate( wtab_ed(numlep,ntemp,nedens) )
      allocate( wtab_pc(numlep,ntemp,nedens) )
      allocate( wtab_edpcnu(numlep,ntemp,nedens) )
c     fill compact tables
      do edensi=1,nedens
       do tempi=1,ntemp
        do i=1,numnuc
         delepi = map_nuci_to_delepi(i)
         if (delepi .gt. 0 ) then
          wtab_pd(delepi,tempi,edensi) = s_wtab_pd(i,tempi,edensi)
          wtab_ec(delepi,tempi,edensi) = s_wtab_ec(i,tempi,edensi)
          wtab_pdecnu(delepi,tempi,edensi)=s_wtab_pdecnu(i,tempi,edensi)
         endif
         lepi = map_nuci_to_lepi(i)
         if (lepi .gt. 0 ) then
          wtab_ed(lepi,tempi,edensi) = s_wtab_ed(i,tempi,edensi)
          wtab_pc(lepi,tempi,edensi) = s_wtab_pc(i,tempi,edensi)
          wtab_edpcnu(lepi,tempi,edensi) =s_wtab_edpcnu(i,tempi,edensi)
         endif
        enddo
       enddo
      enddo

c     deallocate sparse tables and info arrays
      deallocate( s_wtab_pd )
      deallocate( s_wtab_ec )
      deallocate( s_wtab_pdecnu )
      deallocate( s_wtab_ed )
      deallocate( s_wtab_pc )
      deallocate( s_wtab_edpcnu )


c     test table
c     find ti45
c      do i=1,numnuc
c       if (zion(i)==22 .and. aion(i)==45) exit
c      enddo
c      print *, 'ti45 nuci: ', i
c      i = map_nuci_to_delepi(i)
c      print *, 'ti45 values: ',wtab_pd(i,6,3), wtab_ec(i,6,3)
c      stop

      end subroutine weaktab_init

c===================================================================

      subroutine nucname2num(nname, z, a)
      implicit none
      character(len=5),intent(in) :: nname
      integer,intent(out)         :: z, a

      integer, parameter     :: zsymbnum=85
      character(len=2)       :: zsymb(zsymbnum), element

      data  zsymb/'h ','he','li','be','b ','c ','n ','o ','f ','ne',
     1            'na','mg','al','si','p ','s ','cl','ar','k ','ca',
     2            'sc','ti','v ','cr','mn','fe','co','ni','cu','zn',
     3            'ga','ge','as','se','br','kr','rb','sr','y ','zr',
     4            'nb','mo','tc','ru','rh','pd','ag','cd','in','sn',
     5            'sb','te','i' ,'xe','cs','ba','la','ce','pr','nd',
     6            'pm','sm','eu','gd','tb','dy','ho','er','tm','yb',
     7            'lu','hf','ta','w' ,'re','os','ir','pt','au','hg',
     8            'tl','pb','bi','po','at'/

c     special cases
      if ( trim(nname)=='p') then
       z=1
       a=1
       return
      endif
      if ( trim(nname)=='n') then
       z=0
       a=1
       return
      endif

c     split name into element and baryon number
c     element name is either 2 or 1 characters
      if ( (ichar(nname(2:2)).ge.ichar('a'))
     1                  .and.(ichar(nname(2:2)).le.ichar('z')) ) then
c      2 character element name
       element = nname(1:2)
       read(nname(3:5),*) a
      else
c      1 character element name
       element(1:1)=nname(1:1)
       element(2:2)=' '
       read(nname(2:4),*) a
      endif

      do z=1,zsymbnum
       if ( element .eq. zsymb(z) ) exit
      enddo

      if ( z .gt. zsymbnum) then
       print *, 'unable to find element name: ', element
       stop
      endif

      end subroutine nucname2num


c===================================================================
c     subroutine to fill the relevant entries in the rate arrays for
c     use in the main torch routines.  The e^+/- decay rates from
c     reaclib are assumed to already be put in place, so if we don't
c     have a rate for a particular target nuclide, we should leave it
c     alone.

c     We pass the full rate arrays to keep the compiler from doing
c     any unnecessary copying.  We will only be setting the entries
c     r_tab(5,:) and r_tab(6,:)
c     r_tab(5,i) is the leptonization rate for target nuclide i per nuclei i
c     r_tab(6,i) is the rate at which nuclide i is produced by deleptonization
c                    per parent target nuclei k
      subroutine weaktab_rates(dens,temp,ye,
     1               r_np, drdt_np, drdd_np,
     2               r_pn, drdt_pn, drdd_pn,
     3               r_tab, drdt_tab, drdd_tab)


      implicit none
      double precision, intent(in) :: dens, temp, ye
      double precision, intent(inout) :: r_np, drdt_np, drdd_np,
     1                                    r_pn, drdt_pn, drdd_pn,
     2                     r_tab(:,:), drdt_tab(:,:), drdd_tab(:,:)

      integer :: tempi, ledensi, i, lepi, delepi, ihi, ilo, prodi
      double precision :: ledens, tw, dw, dt_inv, lnten
      double precision :: dlrdt_ed, dlrdled_ed, r_ed, drdt_ed, drdd_ed
      double precision :: dlrdt_pc, dlrdled_pc, r_pc, drdt_pc, drdd_pc
      double precision :: dlrdt_pd, dlrdled_pd, r_pd, drdt_pd, drdd_pd
      double precision :: dlrdt_ec, dlrdled_ec, r_ec, drdt_ec, drdd_ec

      lnten=log(10.0e0)
      ledens = log10(ye*dens)
c     cut out early if we are out of bounds
c     we leave the rate tables unchanged, in case they were filled
c     by another routine
      if ( temp < tempgrid(1) .or. temp > tempgrid(ntemp) .or.
     1     ledens < ledens_min .or. ledens > ledens_max ) return

c     ---------- Determine indices and weighting factors
      ledensi = min(nedens-1,int(floor(ledens)-ledens_min)+1)

c     Check if saved tempi is still valid
      tempi = tempi_saved
      if ( temp<tempgrid(tempi) .or. temp>=tempgrid(tempi+1) ) then
c      re-find temp index via binary search
       ilo = 1
       ihi = ntemp-1
       do while (ihi.ne.ilo)
        tempi = ilo+(ihi-ilo+1)/2
        if ( temp<tempgrid(tempi) ) then
         ihi = tempi-1
        else
         ilo = tempi
        endif
       enddo
      endif
      tempi_saved = tempi
       
c     Calculate weighting factors for linear interpolation
c     these have units since they will be multiplied by derivatives
      tw = temp-tempgrid(tempi)
      dw = ledens-ledensi
      dt_inv = 1.0e0/(tempgrid(tempi+1)-tempgrid(tempi))

c     -----------  Step through all nuclides and fill rate structures
c     rates are left unchanged if we don't have an entry
      do i=1,numnuc

c  First do leptonizing direction
       lepi = map_nuci_to_lepi(i)
       if (lepi > 0) then
c       interpolate rates and derivatives
        dlrdt_ed=(wtab_ed(lepi,tempi+1,ledensi)
     1                                     -wtab_ed(lepi,tempi,ledensi))
     2            *dt_inv
        dlrdled_ed=(wtab_ed(lepi,tempi,ledensi+1)
     1                                     -wtab_ed(lepi,tempi,ledensi))
        r_ed=10**(wtab_ed(lepi,tempi,ledensi)+tw*dlrdt_ed+dw*dlrdled_ed)
        drdt_ed = lnten*r_ed*dlrdt_ed
        drdd_ed = r_ed/dens*dlrdled_ed

        dlrdt_pc=(wtab_pc(lepi,tempi+1,ledensi)
     1                                     -wtab_pc(lepi,tempi,ledensi))
     2            *dt_inv
        dlrdled_pc=(wtab_pc(lepi,tempi,ledensi+1)
     1                                     -wtab_pc(lepi,tempi,ledensi))
        r_pc=10**(wtab_pc(lepi,tempi,ledensi)+tw*dlrdt_pc+dw*dlrdled_pc)
        drdt_pc = lnten*r_pc*dlrdt_pc
        drdd_pc = r_pc/dens*dlrdled_pc

c       proton & neutron rates are stored separately
        if (i.eq.nnuci) then
         r_np = r_ed+r_pc
         drdt_np = drdt_ed+drdt_pc
         drdd_np = drdd_ed+drdd_pc
        else
         r_tab(5,i) = r_ed+r_pc
         drdt_tab(5,i) = drdt_ed+drdt_pc
         drdd_tab(5,i) = drdd_ed+drdd_pc
        endif

       endif

c  Now do deleptonizing direction
       delepi = map_nuci_to_delepi(i)
       if (delepi > 0) then
c       interpolate rates and derivatives
        dlrdt_pd=(wtab_pd(delepi,tempi+1,ledensi)
     1                                   -wtab_pd(delepi,tempi,ledensi))
     2            *dt_inv
        dlrdled_pd=(wtab_pd(delepi,tempi,ledensi+1)
     1                                   -wtab_pd(delepi,tempi,ledensi))
        r_pd=10**(wtab_pd(delepi,tempi,ledensi)
     1                                       +tw*dlrdt_pd+dw*dlrdled_pd)
        drdt_pd = lnten*r_pd*dlrdt_pd
        drdd_pd = r_pd/dens*dlrdled_pd

        dlrdt_ec=(wtab_ec(delepi,tempi+1,ledensi)
     1                                   -wtab_ec(delepi,tempi,ledensi))
     2            *dt_inv
        dlrdled_ec=(wtab_ec(delepi,tempi,ledensi+1)
     1                                   -wtab_ec(delepi,tempi,ledensi))
        r_ec=10**(wtab_ec(delepi,tempi,ledensi)
     1                                       +tw*dlrdt_ec+dw*dlrdled_ec)
        drdt_ec = lnten*r_ec*dlrdt_ec
        drdd_ec = r_ec/dens*dlrdled_ec

c       proton & neutron rates are stored separately
        if (i.eq.pnuci) then
         r_pn = r_pd+r_ec
         drdt_pn = drdt_pd+drdt_ec
         drdd_pn = drdt_pd+drdt_ec
        else
         prodi = delep_prod(i)
         r_tab(6,prodi) = r_pd+r_ec
         drdt_tab(6,prodi) = drdt_pd+drdt_ec
         drdd_tab(6,prodi) = drdd_pd+drdd_ec
        endif

       endif
c      to spot check
c       if (i==154) then
c        print *,'log10 ni56 pd ec ed pc rates:',
c     1     log10(r_pd), log10(r_ec), log10(r_ed), log10(r_pc)
c        print *,lepi,delepi
c       endif

      enddo

      end subroutine

      end module weaktab
