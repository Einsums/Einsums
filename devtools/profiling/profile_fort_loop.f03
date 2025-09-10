MODULE loops
   IMPLICIT NONE

CONTAINS
!       Create the J matrix.
   SUBROUTINE create_j(j, d, tei, norbs)
      REAL*8, DIMENSION(:,:), INTENT(OUT) :: j
      REAL*8, DIMENSION(:,:), INTENT(IN) :: d
      REAL*8, DIMENSION(:,:,:,:), INTENT(IN) :: tei

      INTEGER, INTENT(IN) :: norbs


      INTEGER :: mu, nu, lam, sig

      j(:,:) = 0

      DO CONCURRENT (mu=1:norbs, nu=1:norbs, lam=1:norbs, sig=1:norbs)
         j(nu, mu) = j(nu, mu) + 2 * d(nu, mu) * tei(sig, lam, nu, mu)
      END DO

   END SUBROUTINE create_j

! Create the K matrix.
   SUBROUTINE create_k(k, d, tei, norbs)
      REAL*8, DIMENSION(:,:), INTENT(OUT) :: k
      REAL*8, DIMENSION(:,:), INTENT(IN) :: d
      REAL*8, DIMENSION(:,:,:,:), INTENT(IN) :: tei

      INTEGER, INTENT(IN) :: norbs


      INTEGER :: mu, nu, lam, sig

      k(:,:) = 0

      DO CONCURRENT (mu=1:norbs, nu=1:norbs, lam=1:norbs, sig=1:norbs)
         k(nu, mu) = k(nu, mu) - d(nu, mu) * tei(sig, nu, lam, mu)
      END DO

   END SUBROUTINE create_k

! Create the G matrix.
   SUBROUTINE create_g(g, j, k)
      REAL(8), DIMENSION(:,:), INTENT(OUT) :: g
      REAL(8), DIMENSION(:,:), INTENT(IN) :: j, k

      g(:,:) = j(:,:) + k(:,:)
   END SUBROUTINE

! Calculate the average.
   REAL(8) FUNCTION mean(values)
      REAL(8), DIMENSION(:), INTENT(IN) :: values

      mean = SUM(values) / SIZE(values)
   END FUNCTION mean

! Calculate the variance
   REAL(8) FUNCTION variance(values, avg)
      REAL(8), DIMENSION(:), INTENT(IN) :: values
      REAL(8), INTENT(IN) :: avg

      INTEGER :: i

      variance = 0

      DO i=1,SIZE(values)
         variance = variance + (values(i) - avg) * (values(i) - avg)
      END DO

      variance = variance / (SIZE(values) - 1)
   END FUNCTION variance

! Calculate the standard deviation.
   REAL(8) FUNCTION stdev(values, avg)
      REAL(8), DIMENSION(:), INTENT(IN) :: values
      REAL(8), INTENT(IN) :: avg

      stdev = SQRT(variance(values, avg))
   END FUNCTION stdev

! Parse command line arguments
   SUBROUTINE parse_args(norbs, trials)
      INTEGER, INTENT(OUT) :: norbs, trials

      INTEGER :: i, state, status
      CHARACTER(len=64) :: arg

      state = 0
      status = 0
      norbs = 20
      trials = 20

      DO i = 1, COMMAND_ARGUMENT_COUNT()
         CALL GET_COMMAND_ARGUMENT(i, arg)
         SELECT CASE(state)
          CASE (0)
            IF(arg == "-n") THEN
               state = 1
            ELSEIF(arg == "-t") THEN
               state = 2
            ELSEIF(arg == "-h" .OR. arg == "--help") THEN
               WRITE(*,*) "Arguments:\n\n&
               &-n NUMBER\t\tThe number of orbitals. Defaults to 20.\n\n&
               &-t NUMBER\t\tThe number of trials. Defaults to 20.\n\n-h, --help\t\tPrint the help message."
            ELSE
               STOP "Error! Could not handle argument. Try -h or --help for help. &
               &Also check to make sure there are spaces between your arguments."
            END IF
          CASE (1)
            READ(arg, *, iostat=state) norbs

            IF(state /= 0) THEN
               STOP "Could not handle integer argument! Try -h or --help for help."
            ELSEIF(norbs < 1) THEN
               STOP "Invalid number of orbitals. Number of orbitals must be greater than 0."
            ENDIF
          CASE (2)
            READ(arg, *, iostat=state) trials

            IF(state /= 0) THEN
               STOP "Could not handle integer argument! Try -h or --help for help."
            ELSEIF(trials < 1) THEN
               STOP "Invalid number of trials. Number of trials must be greater than 0."
            ENDIF
          CASE DEFAULT
            STOP "Something really bad happened."
         END SELECT
      END DO
   END SUBROUTINE parse_args

END MODULE loops

PROGRAM time_loops
   USE loops

   IMPLICIT NONE

   INTEGER :: norbs, trials
   REAL(8), ALLOCATABLE, DIMENSION(:,:) :: J, K, D, G
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:,:) :: TEI
   REAL(8), ALLOCATABLE, DIMENSION(:) :: time_J, time_K, time_G, time_tot
   REAL(8) :: mean_J, mean_K, mean_G, mean_tot, start, J_split, K_split, G_split

   INTEGER :: i

   CALL parse_args(norbs, trials)

   PRINT *, "Running ", trials, " trials with ", norbs, " orbitals."

   CALL RANDOM_INIT(.FALSE., .FALSE.)

   ALLOCATE(J(norbs, norbs), K(norbs, norbs), D(norbs, norbs), G(norbs, norbs))
   ALLOCATE(TEI(norbs, norbs, norbs, norbs))
   ALLOCATE(time_J(trials), time_K(trials), time_G(trials), time_tot(trials))

! Initialize the input arrays.
    CALL RANDOM_NUMBER(D)
    CALL RANDOM_NUMBER(TEI)

    D = 2 * D - 1
    TEI = 2 * TEI - 1

! Perform the trials
    DO i=1,trials
        CALL CPU_TIME(start)
        CALL create_j(J, D, TEI, norbs)
        CALL CPU_TIME(J_split)
        CALL create_k(K, D, TEI, norbs)
        CALL CPU_TIME(K_split)
        CALL create_g(G, J, K)
        CALL CPU_TIME(G_split)

        time_J(i) = J_split - start
        time_tot(i) = G_split - start
        time_K(i) = K_split - J_split
        time_G(i) = G_split - K_split
    END DO

    DEALLOCATE(J, K, D, G, TEI)

    mean_J = mean(time_J)
    mean_K = mean(time_K)
    mean_G = mean(time_G)
    mean_tot = mean(time_tot)

    PRINT *, "Timing information:"
    PRINT *, "Form J: ", mean_J, " s, stdev ", stdev(time_J, mean_J), " s"
    PRINT *, "Form K: ", mean_K, " s, stdev ", stdev(time_K, mean_K), " s"
    PRINT *, "Form G: ", mean_G, " s, stdev ", stdev(time_G, mean_G), " s"
    PRINT *, "Total: ", mean_tot, " s, stdev ", stdev(time_tot, mean_tot), " s"

    DEALLOCATE(time_J, time_K, time_G, time_tot)

END PROGRAM time_loops
